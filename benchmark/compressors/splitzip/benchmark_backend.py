"""Frozen dataset calibration and merged official-kernel SplitZip batch decoding."""
from adapters.benchmark_cpu import CPUBackend
from adapters.nvcomp_benchmark import upload
from benchmark_routes import module_at
from adapters.paper_support import packet,unpacket


class Backend(CPUBackend):
    def __init__(self,route):
        super().__init__(route)
        import torch
        self.torch=torch;self.stream=torch.cuda.Stream();self.api=module_at(route.folder/'api.py')
        self.fp8=str(route.dtype).startswith('fp8')
        if self.fp8:
            self.fmt=self.api.FP8_FORMATS[route.dtype];self.bits=self.api.code_bits(route.options)
            self.module=self.api.upstream_fp8()
            self.codec=self.module.SplitZipFP8GPU(self.fmt,self.bits,device='cuda:0')
            self.chunk=self.api.FP8_CHUNK
        else:
            self.module=self.api.upstream()
            self.chunk=int(route.options.get('chunk_size',1024))
            self.codec=self.module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=self.chunk)

    def setup(self,sample):
        if self.fp8:
            tensor=self.torch.frombuffer(bytearray(sample),dtype=self.torch.uint8).to('cuda')
            with self.torch.cuda.stream(self.stream):self.coverage=self.codec.calibrate(tensor)
            self.stream.synchronize();self.book=list(self.codec.codebook)
            return
        with self.torch.cuda.stream(self.stream):
            tensor=self.torch.frombuffer(bytearray(sample),dtype=self.torch.bfloat16).to('cuda')
            self.coverage=self.codec.calibrate(tensor)
        self.stream.synchronize();self.book=self.codec.dec_lut.cpu().tolist()

    def prepare_encode(self,raw):
        if self.fp8:
            with self.torch.cuda.stream(self.stream):return self.torch.frombuffer(bytearray(raw),dtype=self.torch.uint8).to('cuda')
        with self.torch.cuda.stream(self.stream):return self.torch.frombuffer(bytearray(raw),dtype=self.torch.bfloat16).to('cuda')

    def encode(self,tensor):
        with self.torch.cuda.stream(self.stream):return self.codec.encode(tensor)

    def finish_encode(self,value,size):
        if self.fp8:
            parts=[self.api.tensor_bytes(value.p32)]
            if value.p16 is not None:parts.append(self.api.tensor_bytes(value.p16))
            if value.p8 is not None:parts.append(self.api.tensor_bytes(value.p8))
            parts.append(self.api.tensor_bytes(value.counts.to(self.torch.int16)))
            parts.append(self.api.tensor_bytes(value.esc))
            parts.append(self.api.tensor_bytes(value.tail))
            header=dict(format='splitzip-fp8-packet-v1',fmt=self.fmt,code_bits=self.bits,n=value.n,n_esc=value.n_esc,
                        codebook=list(self.book),p16=value.p16 is not None,p8=value.p8 is not None,
                        calibration='frozen first page; excluded from codec timing',coverage=self.coverage)
            return packet(header,parts)
        header=dict(format='splitzip-packet-v1',n=value.n,n_esc=value.n_esc,chunk_size=value.chunk_size,codebook=self.book,calibration='frozen first page; excluded from codec timing',coverage=self.coverage)
        return packet(header,[self.api.tensor_bytes(getattr(value,k)) for k in ('pk','sm','chunk_id','local_pos','esc_val')])

    def prepare_decode(self,frames,sizes):
        if len(frames)!=len(sizes):raise ValueError('SplitZip frame/size count mismatch')
        if self.fp8:return self._prepare_decode_fp8(frames,sizes)
        torch=self.torch;packets=[self.api.decode_packet(f,size) for f,size in zip(frames,sizes)];buffers=[]
        for (h,parts),size in zip(packets,sizes):
            n=h['n'];esc=h['n_esc']
            if h['format']!='splitzip-packet-v1' or n*2!=size or not 0<=esc<=n or len(parts)!=5 or [len(p) for p in parts]!=[(n+1)//2,n,4*esc,2*esc,esc]:raise ValueError('Invalid SplitZip frame')
            if h['codebook']!=self.book or h['chunk_size']!=self.chunk:raise ValueError('SplitZip batch codebook/chunk mismatch')
            buffers.extend(parts)
        # Padding is outside timing; it permits arbitrary tail pages while retaining
        # every page's independent escape positions and a single official decode call.
        host,device,offsets=upload(buffers);pk=[];sm=[];cid=[];pos=[];escapes=[];segments=[];base=0
        for i,((h,parts),size) in enumerate(zip(packets,sizes)):
            views=[device[o:o+len(p)] for o,p in zip(offsets[i*5:i*5+5],parts)]
            n=h['n'];padded=(n+self.chunk-1)//self.chunk*self.chunk
            pk.append(torch.nn.functional.pad(views[0],(0,padded//2-len(parts[0]))))
            sm.append(torch.nn.functional.pad(views[1],(0,padded-n)))
            cid.append(views[2].view(torch.int32)+base//self.chunk);pos.append(views[3]);escapes.append(views[4]);segments.append((base*2,size));base+=padded
        merged=self.module.ChunkLocalGPUEncoded(torch.cat(pk),torch.cat(sm),torch.empty(0,dtype=torch.int32,device='cuda'),torch.empty(0,dtype=torch.int64,device='cuda'),torch.cat(cid),torch.cat(pos).view(torch.uint16),torch.cat(escapes),base,sum(h['n_esc'] for h,p in packets),self.chunk)
        return dict(call=lambda:self.codec.decode(merged).view(torch.uint8),finish=lambda out:[out[start:start+size] for start,size in segments],keep=(host,device,merged))

    def _prepare_decode_fp8(self,frames,sizes):
        torch=self.torch;packets=[self.api.decode_packet(f,size) for f,size in zip(frames,sizes)]
        first=packets[0][0];has16=bool(first['p16']);has8=bool(first['p8'])
        for (h,parts),size in zip(packets,sizes):
            if h.get('format')!='splitzip-fp8-packet-v1':raise ValueError('Invalid SplitZip FP8 frame')
            if h['n']!=size:raise ValueError('Invalid SplitZip FP8 size')
            if list(h['codebook'])!=self.book:raise ValueError('SplitZip batch codebook mismatch')
            if bool(h['p16'])!=has16 or bool(h['p8'])!=has8:raise ValueError('SplitZip batch plane mismatch')
        i_counts=1+int(has16)+int(has8);i_esc=i_counts+1;i_tail=i_esc+1

        def buf(value,dtype):
            return torch.frombuffer(bytearray(value),dtype=dtype).to('cuda') if value else torch.empty(0,dtype=dtype,device='cuda')

        def stack(index,dtype):
            items=[buf(p[index],dtype) for h,p in packets]
            total=sum(x.numel() for x in items)
            if not total:return torch.empty(0,dtype=dtype,device='cuda')
            out=torch.empty(total,dtype=dtype,device='cuda')
            off=0
            for x in items:
                if x.numel():out[off:off+x.numel()]=x;off+=x.numel()
            return out

        p32=stack(0,torch.int32)
        p16=stack(1,torch.int16) if has16 else None
        p8=stack(1+int(has16),torch.uint8) if has8 else None
        counts=stack(i_counts,torch.int16).to(torch.int32)
        esc=stack(i_esc,torch.int16)
        tails=[buf(p[i_tail],torch.uint8) for h,p in packets]
        nch=[self.api.nch_for(h['n']) for h,_ in packets]
        body=[value*self.chunk for value in nch]
        offsets=[];total=0
        for value in body:
            offsets.append(total);total+=value
        starts=(torch.cumsum(counts,0)-counts).to(torch.int32) if counts.numel() else torch.empty(0,dtype=torch.int32,device='cuda')
        empty=torch.empty(0,dtype=torch.uint8,device='cuda')
        merged=self.module.FP8Encoded(p32,p16,p8,counts,starts,esc,empty,total,esc.numel())

        def split(out):
            results=[]
            for off,length,tail in zip(offsets,body,tails):
                part=out[off:off+length]
                results.append(torch.cat([part,tail]) if tail.numel() else part)
            return results

        return dict(call=lambda:self.codec.decode(merged),finish=split,keep=(merged,))

    def decode_signature(self,sizes):return tuple(sizes)

    def validate_decoded(self,parts,prepared):pass
