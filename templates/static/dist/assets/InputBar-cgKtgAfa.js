import{i as n,d as _,l as z,c as f,a,D as I,E as V,f as m,u as o,F as w,b as F,p as y,g as k,o as d,n as B,e as S,G as E,_ as j}from"./index-B6SwdMa1.js";import{u as q}from"./chat-CmItOQ5N.js";import{F as D}from"./file-text-C71GGb-A.js";import{C as A}from"./chart-column-nX5WSeem.js";/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const H=n("cpu",[["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M17 20v2",key:"1rnc9c"}],["path",{d:"M17 2v2",key:"11trls"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M2 17h2",key:"7oei6x"}],["path",{d:"M2 7h2",key:"asdhe0"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"M20 17h2",key:"1fpfkl"}],["path",{d:"M20 7h2",key:"1o8tra"}],["path",{d:"M7 20v2",key:"4gnj0m"}],["path",{d:"M7 2v2",key:"1i4yhu"}],["rect",{x:"4",y:"4",width:"16",height:"16",rx:"2",key:"1vbyd7"}],["rect",{x:"8",y:"8",width:"8",height:"8",rx:"1",key:"z9xiuo"}]]);/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const K=n("flask-conical",[["path",{d:"M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2",key:"18mbvz"}],["path",{d:"M6.453 15h11.094",key:"3shlmq"}],["path",{d:"M8.5 2h7",key:"csnxdl"}]]);/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const L=n("message-square",[["path",{d:"M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z",key:"18887p"}]]);/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const N=n("paperclip",[["path",{d:"m16 6-8.414 8.586a2 2 0 0 0 2.829 2.829l8.414-8.586a4 4 0 1 0-5.657-5.657l-8.379 8.551a6 6 0 1 0 8.485 8.485l8.379-8.551",key:"1miecu"}]]);/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const T=n("send",[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]]),G={class:"input-bar"},P={class:"input-row"},R=["placeholder","disabled"],U=["disabled"],$={class:"input-toolbar"},J=["title","onClick"],O=_({__name:"InputBar",props:y({disabled:{type:Boolean},placeholder:{}},{modelValue:{default:""},modelModifiers:{}}),emits:y(["send","fileSelected"],["update:modelValue"]),setup(i,{emit:v}){const c=q(),s=z(i,"modelValue"),r=v,h=k(),u=k(),b=[{id:"normal",label:"对话",icon:L,hint:"自由对话"},{id:"extraction",label:"文献提取",icon:D,hint:'输入"帮我搜寻：<描述>"开始提取'},{id:"hardware",label:"硬件控制",icon:H,hint:'输入"硬件控制：<指令>"操控设备'},{id:"experiment",label:"实验设计",icon:K,hint:'输入"实验设计：<描述>"设计实验'},{id:"analysis",label:"数据分析",icon:A,hint:'输入"数据分析"开始分析'}];function x(e){e.key==="Enter"&&!e.shiftKey&&(e.preventDefault(),p())}function p(){const e=s.value.trim();e&&(r("send",e),s.value="")}function M(){const e=h.value;e&&(e.style.height="auto",e.style.height=e.scrollHeight+"px")}function g(){var e;(e=u.value)==null||e.click()}function C(e){const t=e.target;t.files&&t.files.length>0&&(r("fileSelected",t.files[0]),t.value="")}return(e,t)=>(d(),f("div",G,[a("div",P,[I(a("textarea",{ref_key:"textarea",ref:h,"onUpdate:modelValue":t[0]||(t[0]=l=>s.value=l),class:"input-textarea",placeholder:i.placeholder||"输入消息... (Enter 发送, Shift+Enter 换行)",disabled:i.disabled,rows:"1",onKeydown:x,onInput:M},null,40,R),[[V,s.value]]),a("button",{class:"send-btn",disabled:i.disabled||!s.value.trim(),onClick:p},[m(o(T),{size:18})],8,U)]),a("div",$,[a("input",{ref_key:"fileInput",ref:u,type:"file",accept:".pdf,.csv,.txt,.json,.xlsx,.xls",class:"file-input-hidden",onChange:C},null,544),a("button",{class:"toolbar-btn",title:"上传文件",onClick:g},[m(o(N),{size:15})]),t[1]||(t[1]=a("span",{class:"toolbar-divider"},null,-1)),(d(),f(w,null,F(b,l=>a("button",{key:l.id,class:B(["toolbar-btn",{active:o(c).currentMode===l.id}]),title:l.hint,onClick:Q=>o(c).setMode(l.id)},[(d(),S(E(l.icon),{size:16}))],10,J)),64))])]))}}),ee=j(O,[["__scopeId","data-v-c912b717"]]);export{H as C,K as F,ee as I};
