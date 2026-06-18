import{c as r}from"./index-DO4N0L_Q.js";/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const o=r("chart-column",[["path",{d:"M3 3v16a2 2 0 0 0 2 2h16",key:"c24i48"}],["path",{d:"M18 17V9",key:"2bz60n"}],["path",{d:"M13 17V5",key:"1frdt8"}],["path",{d:"M8 17v-3",key:"17ska0"}]]);async function i(){return(await(await fetch("/api/list_algorithms")).json()).algorithms||[]}async function c(t){return(await fetch("/api/browse_csv")).json()}async function p(t,a,n={}){return(await fetch("/api/run_algorithm",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({algorithm_name:t,input_file:a,params:n})})).json()}async function h(t,a=20){const n=new URLSearchParams({path:t,n:String(a)});return(await fetch(`/api/csv/preview?${n.toString()}`)).json()}async function m(t){return(await fetch("/api/algorithm/recommend",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:t})})).json()}export{o as C,p as a,c as b,i as l,h as p,m as r};
