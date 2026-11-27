
const rows=__ROWS__;
const csv=__CSV__; const omitted=__UNAVAILABLE__;document.querySelector('#unavailable').textContent=Object.entries(omitted).map(([name,reason])=>name+': '+reason).join('\n')||'无 / None';
const metric=document.querySelector('#metric'),route=document.querySelector('#route'),table=document.querySelector('#heatmap');
const decoder=document.querySelector('#decoder');
const storage=document.querySelector('#storage');
for(const dtype of new Set(rows.map(r=>r.storage_dtype||'bf16'))){const option=document.createElement('option');option.value=dtype;option.textContent=({bf16:'BF16',fp8_e4m3fn:'FP8 E4M3FN',fp8_e5m2:'FP8 E5M2'})[dtype]||dtype;storage.append(option)}
const toGBs = value => value * 1048576 / 1e9;
const formatGBs = value => Number.isFinite(value) ? toGBs(value).toFixed(3)+' GB/s' : '—';
function describeBatch(value, fallback, name){
if(!value)return name+' '+(fallback||'?')+' (\u914d\u7f6e\uff0c\u5b9e\u6d4b\u6279\u91cf\u672a\u77e5)';
let text=name+' '+value.regular;
if(value.tail!==null)text+='\uff08\u672b\u6b21 '+value.tail+'\uff09';
if(value.irregular)text+=' \u00b7 \u4e2d\u9014\u4e5f\u6709\u4e0d\u8db3\u6574\u6279';
return text;
}
function batchLabel(row){
return describeBatch(row.encode_batch_display,row.encode_batch_requested_blocks,'\u7f16\u7801\u6279')+' \uff5c '+describeBatch(row.decode_batch_display,row.decode_batch_blocks,'\u89e3\u7801\u6279');
}
function render(){
const selected=rows.filter(r=>(route.value==='all'||r.execution_path===route.value)&&(storage.value==='all'||(r.storage_dtype||'bf16')===storage.value)),data=selected.filter(r=>r.execution_path.split('-').pop()===decoder.value && Number.isFinite(r[metric.value])),names=[...new Set(data.map(r=>r.codec))],datasets=[...new Set(selected.map(r=>r.dataset))];
const finite=data.map(r=>r[metric.value]).filter(Number.isFinite),max=finite.length?Math.max(...finite):0,min=finite.length?Math.min(...finite):0;
table.replaceChildren();
if (!names.length) { table.createCaption().textContent='当前筛选没有实测数值'; document.querySelector('#scale').textContent=''; return; }const head=table.createTHead().insertRow();let h=document.createElement('th');h.textContent='数据集 / Codec';head.append(h);
for(const name of names){const th=document.createElement('th');th.textContent=name;const small=document.createElement('small');const info=data.find(r=>r.codec===name)||rows.find(r=>r.codec===name);small.textContent=(data.some(r=>r.codec===name)?info.execution_path:'/')+' · 编码批量（页/帧）· 解码批量（页/批）';th.append(small);th.title=rows.find(r=>r.codec===name).codec_note||'';head.append(th)}
const body=table.createTBody();for(const dataset of datasets){const tr=body.insertRow(),th=document.createElement('th');th.textContent=dataset;const small=document.createElement('small');small.textContent=(rows.find(r=>r.dataset===dataset).block_bytes/1024)+' KiB / 块';th.append(small);tr.append(th);
for(const name of names){const r=data.find(r=>r.dataset===dataset&&r.codec===name),td=tr.insertCell();if(r&&!Number.isFinite(r[metric.value])){td.textContent='/';continue;}if(!r){td.textContent='/';td.title='该解码端没有对应方案或实测结果';continue;}const v=r[metric.value],t=max>min?(v-min)/(max-min):0.5;td.textContent=metric.value==='compression_ratio'?v.toFixed(2)+'×':toGBs(v).toFixed(3);td.style.background=`hsl(${205-37*t} ${40+35*t}% ${96-58*t}%)`;td.style.color=(96-58*t)<58?'#ffffff':'#162638';td.tabIndex=0;const batch=document.createElement('small');batch.className='batch-info';batch.textContent=batchLabel(r);td.append(batch);
let text=`${dataset} · ${name} · ${r.execution_path}
${batchLabel(r)}
压缩比：${r.compression_ratio.toFixed(4)}× ｜ 编码：${formatGBs(r.encode_mib_s)} ｜ 解码：${formatGBs(r.decode_mib_s)}
原始：${r.raw_bytes.toLocaleString()} 字节 ｜ 压缩后：${r.compressed_bytes.toLocaleString()} 字节 ｜ 无损校验：${r.verified_blocks.toLocaleString()} / ${r.blocks.toLocaleString()} 块`;
if(r.encode_batch_policy||r.decode_batch_policy)text+='\nBatch policy: encode='+r.encode_batch_policy+'; decode='+r.decode_batch_policy;
if(r.batch_selection)text+='\nBatch selection: '+r.batch_selection;
if(r.encode_batch_requested_blocks)text+='\n编码配置：'+r.encode_batch_requested_blocks+' 页/帧；单元格显示实测批量与末次批量。';
if(r.storage_dtype)text+='\n存储格式：'+r.storage_dtype+' ｜ 页面格式：'+r.page_format+' ｜ data/scale 字节：'+r.data_bytes+' / '+r.scale_bytes+' ｜ 量化来源：'+r.quantization_origin;
if(r.quantization_metrics&&r.quantization_metrics!=='{}')text+='\n量化误差元数据：'+r.quantization_metrics;
if(r.codec_note)text+='\nCodec 备注：'+r.codec_note;text+='\n计时口径：'+r.timing_note;td.title=text;td.setAttribute('aria-label',text);for(const e of ['mouseenter','focus','click'])td.addEventListener(e,()=>document.querySelector('#detail').textContent=text);
}}
document.querySelector('#scale').textContent=`当前色标范围：${metric.value==='compression_ratio'?min.toFixed(2)+' ×':formatGBs(min)} — ${metric.value==='compression_ratio'?max.toFixed(2)+' ×':formatGBs(max)}。`;
}metric.onchange=route.onchange=decoder.onchange=storage.onchange=render;render();
document.querySelector('#download').onclick=()=>{const url=URL.createObjectURL(new Blob(['﻿'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='result.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
