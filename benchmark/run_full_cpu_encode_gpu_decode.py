"""Generic benchmark with KV pages, shared encoding and fingerprinted JSONL resume."""
from pathlib import Path
import argparse,concurrent.futures,configparser,csv,hashlib,importlib.metadata as md,json,os,sys,time
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parent/'compression_framework')]
from benchmark_routes import select_routes, discover_routes
from codec_timing import Encoder,TIMING_VERSION
from batch_decode import BatchDecoder
from kv_pages import Pages,dataset_configs,validation_subset

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def fingerprint(spec):return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_journal(path):
    done={};encodes={}
    if not path.exists():return done,encodes
    data=path.read_bytes();offset=0
    for line in data.splitlines(keepends=True):
        try:batch=json.loads(line)
        except json.JSONDecodeError:
            if offset+len(line)==len(data) and not line.endswith(b'\n'):
                # Preserve the damaged bytes, then repair only the task's incomplete tail.
                Path(str(path)+'.tail.'+str(time.time_ns())).write_bytes(line)
                with path.open('r+b') as f:f.truncate(offset)
                break
            raise
        if batch.get('type')=='encode':
            for r in batch['records']:encodes[(batch['job'],r['block'])]=r
        else:
            for r in batch['records']:done[(batch['job'],r['block'])]=r
        offset+=len(line)
    if path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:f.write(b'\n')
    return done,encodes
def append(path,kind,job,records):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(type=kind,job=job,records=records),separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

from batch_measure import run_batched_codec
from measurement_identity import layered_spec, decoder_job


def write_rows(out, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = out / 'result.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out / 'result.csv')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    p.add_argument('--validate-only',action='store_true');p.add_argument('--codecs',help='Optional comma-separated validation subset');p.add_argument('--cpu-only',action='store_true');p.add_argument('--settings',type=Path);args=p.parse_args()
    root=args.root.resolve();out=(args.output or root/'benchmark/output/kv_pages_v4_fp8').resolve()
    if not out.is_relative_to(root/'benchmark/output'):raise ValueError('Output must be under benchmark/output')
    out.mkdir(parents=True,exist_ok=True);journal=out/'progress.jsonl';done,encodes=load_journal(journal)
    config=args.settings or root/'benchmark/settings.txt';c=configparser.ConfigParser(interpolation=None);c.read(config)
    from batch_policy import parse_batch, parse_max_bytes, resolve_batches
    workers=c.getint('benchmark','cpu_encode_threads',fallback=1)
    encode_request=parse_batch(c.get('benchmark','encode_batch_blocks',fallback='1'))
    decode_request=parse_batch(c.get('benchmark','decode_batch_blocks',fallback='256'))
    max_bytes=parse_max_bytes(c.get('benchmark','max_batch_bytes',fallback='536870912'))
    memory_fraction=c.getfloat('benchmark','max_batch_memory_fraction',fallback=0.5)
    if not 1<=workers<=256 or not 0<memory_fraction<=0.8:raise ValueError('Invalid workers/batch budget')
    retain_frames=c.getboolean('benchmark','retain_frames',fallback=True)
    import torch
    omitted={}
    common={str(f.relative_to(root)):digest(f) for f in sorted((root/'benchmark/compression_framework').rglob('*.py')) if '__pycache__' not in f.parts}
    common['runner']=digest(__file__);rows=[];capabilities=[]
    for filename,layout in dataset_configs(config).items():
        path=root/'benchmark/input'/filename;pages=Pages(path,layout)
        if args.validate_only:pages=validation_subset(pages)
        count=len(pages)
        dataset_omitted={}
        routes=select_routes(root/'benchmark/compressors',codec_set=c.get('benchmark','codec_set',fallback='core'),execution_device='cpu' if args.cpu_only else c.get('benchmark','execution_device',fallback='cpu'),codecs=args.codecs if args.codecs is not None else c.get('benchmark','codecs',fallback=None),dtype=pages.dtype,skip_incompatible=True,omitted=dataset_omitted)
        if args.cpu_only:routes={name:r for name,r in routes.items() if r.mode=='CPU-CPU'}
        if not routes:
            omitted.update({filename+'::'+name:reason for name,reason in dataset_omitted.items()})
            print('SKIP',filename,'no compatible codecs',flush=True)
            continue
        if c.getboolean('benchmark','skip_unavailable_codecs',fallback=False):
            from runtime_support import unavailable
            for name,route in list(routes.items()):
                reason=unavailable(route,root/'benchmark/compressors')
                if reason:
                    if args.codecs or c.get('benchmark','codecs',fallback='').strip():raise RuntimeError(name+': '+reason)
                    dataset_omitted[name]=reason;del routes[name]
        omitted.update({filename+'::'+name:reason for name,reason in dataset_omitted.items()})
        using_gpu=any(r.mode.endswith('-GPU') for r in routes.values())
        if using_gpu and not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Set an authorized CUDA_VISIBLE_DEVICES')
        environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if using_gpu else 'CPU')
        input_hash=digest(path);sample=pages[0]
        for name,route in routes.items():
            from codec_manager import identity
            installation=identity(name,gpu=route.mode.endswith('-GPU'))
            encode_installation=identity(name,gpu=False) if route.mode=='CPU-GPU' else installation
            runtime=route.runtime();cpu=runtime.cpu_decoder()
            group,batch_size,batch_plan=resolve_batches(runtime,pages,count,encode_request,decode_request,max_bytes,memory_fraction)
            encode_batch=group if encode_request=='max' else encode_request
            if not isinstance(encode_batch,int):encode_batch=1
            print('BATCH_PLAN',filename,name,json.dumps(batch_plan,sort_keys=True),flush=True)
            modes=[route.mode]
            if route.mode.endswith('-GPU') and cpu is not None:modes.append(route.mode.split('-')[0]+'-CPU')
            capabilities.append(dict(dataset=filename,storage_dtype=pages.dtype,codec=name,original=route.mode,cpu_available=cpu is not None,frame_format=route.backend_id,batch_kind=route.options['batch_kind']))
            source={str(f.relative_to(route.folder)):digest(f) for f in sorted(route.folder.rglob('*')) if f.is_file() and f.suffix in ('.py','.json','.so','.dll','.pyd') and '__pycache__' not in f.parts}
            packages={}
            declarations=json.loads((route.folder/'codec.json').read_text())['codecs']
            declaration=next(d for d in declarations if d['name']==name)
            distributions=list(route.options.get('packages',[]))+[declaration.get('distribution','')]
            req=route.folder/'runtime_requirements.json'
            if req.exists():
                from packaging.requirements import Requirement
                distributions += [Requirement(x).name for x in json.loads(req.read_text()).get(name,{}).get('packages',[])]
            for dist in filter(None,distributions):
                try:packages[dist]=md.version(dist)
                except md.PackageNotFoundError:packages[dist]='unavailable'
            spec=dict(installation=installation,encode_installation=encode_installation,storage_dtype=pages.dtype,page_format=pages.page_format,dataset=filename,input_sha256=input_hash,shape=pages.shape,dataset_metadata=pages.dataset_metadata,layout=layout,codec=name,source=source,packages=packages,frame_format=route.backend_id,options=route.options,framework=common,environment=environment,workers=workers,timing=TIMING_VERSION,validation=args.validate_only,count=count,encode_batch=encode_batch if route.mode=='GPU-GPU' else 1)
            layers=layered_spec(root,route,spec,count,encode_batch if route.mode=='GPU-GPU' else 1,group)
            encjob=fingerprint(layers['encode']);jobs={mode:decoder_job(layers,mode,batch_size) for mode in modes}
            spec['batch_plan']=batch_plan
            spec['measurement_layers']=layers
            spec['fingerprint_schema']='layered-measurement-v1'
            for mode,job in jobs.items():(out/('job_'+job+'.json')).write_text(json.dumps(dict(spec,mode=mode,encode_job=encjob),sort_keys=True))
            gpu=BatchDecoder(runtime) if route.mode.endswith('-GPU') else None;cpu_warmed=set();effective=workers if runtime.thread_safe and route.mode.startswith('CPU-') else 1
            pending_work=any((job,i) not in done for job in jobs.values() for i in range(count))
            encoder=Encoder(runtime,sample) if pending_work and not (route.mode == 'GPU-GPU' and encode_batch > 1) else None
            print('JOB',filename,name,modes,'pages',count,'bytes/page',pages.block_bytes,'workers',effective,flush=True)
            if route.mode == 'GPU-GPU' and encode_batch > 1:
                batch_rows=run_batched_codec(out, route, runtime, pages, count, name, filename, layout, jobs, encjob, modes, gpu, batch_size, encode_batch, encoder.setup_seconds if encoder is not None else 0, args.validate_only, retain_frames, group_blocks=group)
                for row in batch_rows:row.update(encode_batch_policy=encode_request,decode_batch_policy=decode_request,batch_selection=json.dumps(batch_plan,sort_keys=True))
                rows.extend(batch_rows)
                write_rows(out, rows)
                continue
            cache=out/'.frames'/encjob;cache.mkdir(parents=True,exist_ok=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective) as pool:
                for start in range(0,count,batch_size):
                    indices=[i for i in range(start,min(start+batch_size,count)) if any((job,i) not in done for job in jobs.values())]
                    if not indices:continue
                    raw=[pages[i] for i in indices]
                    todo=[(i,data) for i,data in zip(indices,raw) if not ((encjob,i) in encodes and (cache/(str(i)+'.bin')).exists())]
                    measurements=dict(zip([i for i,data in todo],encoder.measure_many([data for i,data in todo],pool,effective)))
                    def encode(pair):
                        index,data=pair;old=encodes.get((encjob,index));saved=cache/(str(index)+'.bin')
                        if old and saved.exists():
                            frame=saved.read_bytes()
                            if hashlib.sha256(frame).hexdigest()!=old['frame_sha256']:raise ValueError('Cached frame checksum mismatch')
                            return index,data,frame,old
                        frame,seconds,page_seconds,t0,t1=measurements[index]
                        record=dict(block=index,setup_seconds=encoder.setup_seconds,raw_bytes=len(data),compressed_bytes=len(frame),encode_seconds=seconds,encode_page_seconds=page_seconds,frame_sha256=hashlib.sha256(frame).hexdigest(),page=pages.descriptor(index),started=t0,ended=t1)
                        if old:
                            if old['frame_sha256']!=record['frame_sha256']:raise ValueError('Recreated frame changed; use a fresh experiment')
                            record=old
                        from batch_measure import save_frame
                        save_frame(saved,frame);return index,data,frame,record
                    batch=[encode(pair) for pair in zip(indices,raw)]
                    fresh=[r for i,data,frame,r in batch if (encjob,i) not in encodes]
                    for mode,job in jobs.items():
                        pending=[b for b in batch if (job,b[0]) not in done]
                        if not pending:continue
                        records=[]
                        if mode.endswith('-GPU'):
                            restored,seconds,kind=gpu.submit([b[2] for b in pending],[len(b[1]) for b in pending]).finish();page_s=gpu.page_seconds;total=sum(len(b[1]) for b in pending)
                            decoded=[(b,actual,seconds*len(b[1])/total,page_s*len(b[1])/total,kind) for b,actual in zip(pending,restored)]
                        else:
                            decoded=[]
                            for b in pending:
                                i,data,frame,er=b
                                if len(data) not in cpu_warmed:
                                    if cpu(frame,len(data))!=data:raise ValueError('CPU warmup verification failed')
                                    cpu_warmed.add(len(data))
                                t=time.perf_counter();actual=cpu(frame,len(data));dt=time.perf_counter()-t
                                decoded.append((b,actual,dt,dt,'cpu-single'))
                        for (i,data,frame,er),actual,dt,page_dt,kind in decoded:
                            if actual!=data:raise RuntimeError('Lossless verification failed: '+name+' '+mode+' '+str(i))
                            r=dict(er,decode_seconds=dt,decode_page_seconds=page_dt,decode_batch_kind=kind,encode_job=encjob,actual_batch_blocks=len(pending) if mode.endswith('-GPU') else 1)
                            records.append(r)
                        # A validated decoder makes the shared encoding journal reusable.
                        if fresh:
                            append(journal,'encode',encjob,fresh)
                            for r in fresh:encodes[(encjob,r['block'])]=r
                            fresh=[]
                        append(journal,'decode',job,records)
                        for r in records:done[(job,r['block'])]=r
                    for i in indices:
                        if not retain_frames and all((job,i) in done for job in jobs.values()):(cache/(str(i)+'.bin')).unlink(missing_ok=True)
            if not retain_frames and all((job,i) in done for job in jobs.values() for i in range(count)):
                for i in range(count):(cache/(str(i)+'.bin')).unlink(missing_ok=True)
            if not any(cache.iterdir()):cache.rmdir()
            for mode,job in jobs.items():
                records=[done[(job,i)] for i in range(count)];raw=sum(r['raw_bytes'] for r in records);packed=sum(r['compressed_bytes'] for r in records)
                row=dict(quantization_metrics=json.dumps(pages.dataset_metadata.get('quantization_metrics',{}),sort_keys=True),storage_dtype=pages.dtype,page_format=pages.page_format,quantization_origin=pages.quantization_origin,data_bytes=sum(r['page']['data_bytes'] for r in records),scale_bytes=sum(r['page']['scale_bytes'] for r in records),dataset=path.stem,validation_only=args.validate_only,codec=name,execution_path=mode,job_fingerprint=job,encode_job=encjob,frame_format=route.backend_id,layout=layout['layout'],page_tokens=layout.get('page_tokens',''),block_bytes=pages.block_bytes,cpu_encode_threads=effective,decode_batch_blocks=batch_size if mode.endswith('-GPU') else 1,decode_batch_kind=records[0]['decode_batch_kind'],blocks=count,verified_blocks=count,raw_bytes=raw,compressed_bytes=packed,compression_ratio=raw/packed,encode_mib_s=raw/1048576/sum(r['encode_seconds'] for r in records),decode_mib_s=raw/1048576/sum(r['decode_seconds'] for r in records),encode_page_mib_s=raw/1048576/sum(r['encode_page_seconds'] for r in records),decode_page_mib_s=raw/1048576/sum(r['decode_page_seconds'] for r in records),setup_seconds=records[0].get('setup_seconds',0),timing_version=TIMING_VERSION,fingerprint_schema='layered-measurement-v1',frames_retained=retain_frames,encode_batch_blocks=1,encode_batch_min_blocks=1,encode_batch_requested_blocks=1,encode_timing_scope='gpu_codec_call_cuda_events' if mode.startswith('GPU-') else 'cpu_codec_call_perf_counter',decode_timing_scope='gpu_codec_call_cuda_events' if mode.endswith('-GPU') else 'cpu_codec_call_perf_counter')
                row.update(encode_batch_policy=encode_request,decode_batch_policy=decode_request,batch_selection=json.dumps(batch_plan,sort_keys=True))
                rows.append(row)
            write_rows(out, rows)
    if (out/'.frames').exists() and not any((out/'.frames').iterdir()):(out/'.frames').rmdir()
    (out/'capabilities.json').write_text(json.dumps(capabilities,indent=2))
    if not rows:raise ValueError('No completed eligible configurations')
    (out/'unavailable_codecs.json').write_text(json.dumps(omitted,indent=2))
    from benchmark_engine import write_png
    write_png(out/'heatmap.png',rows,tuple(dict.fromkeys(r['codec'] for r in rows)))
    from html_report import write_report
    write_report(out,None if args.validate_only else root/'benchmark/output/latest.html')
    print('COMPLETE',len(rows),out,flush=True)
if __name__=='__main__':main()
