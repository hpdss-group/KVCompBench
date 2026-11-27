const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Node{
  constructor(){this.children=[];this.textContent='';this.style={};this.value='all';}
  append(...items){this.children.push(...items)}
  replaceChildren(){this.children=[]}
  createTHead(){const n=new Node;this.append(n);return n}
  createTBody(){const n=new Node;this.append(n);return n}
  createCaption(){const n=new Node;this.append(n);return n}
  insertRow(){const n=new Node;this.append(n);return n}
  insertCell(){const n=new Node;this.append(n);return n}
  addEventListener(){}
  setAttribute(){}
}
const nodes={};const get=selector=>nodes[selector]||(nodes[selector]=new Node);
get('#metric').value='encode_mib_s';get('#decoder').value='GPU';
const document={querySelector:get,createElement:()=>new Node};
const rows=[32,256,null].map((batch,i)=>({dataset:'dataset-'+i,codec:'splitzip',execution_path:'GPU-GPU',storage_dtype:'bf16',encode_batch_policy:'max',decode_batch_policy:'max',encode_batch_blocks:batch,encode_batch_requested_blocks:batch,encode_batch_display:batch?{regular:batch,tail:i===1?8:null,irregular:false}:null,decode_batch_display:batch?{regular:256,tail:i===1?8:null,irregular:false}:null,decode_batch_blocks:256,compression_ratio:2,encode_mib_s:100,decode_mib_s:200,block_bytes:1024,raw_bytes:1024,compressed_bytes:512,verified_blocks:1,blocks:1}));
const source=fs.readFileSync(process.argv[2],'utf8').replace('__ROWS__',JSON.stringify(rows)).replace('__CSV__','""').replace('__UNAVAILABLE__','{}');
vm.runInNewContext(source,{document,console,Number,Set,Math,Object});
const table=get('#heatmap'),body=table.children[1];
const labels=body.children.map(row=>row.children[1].children[0].textContent);
assert(labels[0].includes('\u7f16\u7801\u6279 32'));
assert(labels[1].includes('\u7f16\u7801\u6279 256\uff08\u672b\u6b21 8\uff09'));
assert(labels[1].includes('\u89e3\u7801\u6279 256\uff08\u672b\u6b21 8\uff09'));
assert(!labels[0].includes('\u9875/\u5e27'));
assert(labels[2].includes('\u672a\u77e5'));
assert(table.children[0].children[0].children[1].children[0].textContent.includes('\u9875/\u5e27'));
assert(body.children[1].children[1].title.includes('encode=max; decode=max'));
console.log('PASS actual batches, final calls, header units and max policies');
