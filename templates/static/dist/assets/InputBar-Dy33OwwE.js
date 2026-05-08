import{c as r,d as _,p as b,a as v,b as a,G as x,H as y,h as d,u as i,q as c,i as k,o as V,_ as B}from"./index-DjoLPtkD.js";/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const I=r("paperclip",[["path",{d:"m16 6-8.414 8.586a2 2 0 0 0 2.829 2.829l8.414-8.586a4 4 0 1 0-5.657-5.657l-8.379 8.551a6 6 0 1 0 8.485 8.485l8.379-8.551",key:"1miecu"}]]);/**
 * @license lucide-vue-next v1.0.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const M=r("send",[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]]),g={class:"input-bar"},w={class:"input-row"},z=["placeholder","disabled"],E=["disabled"],C={class:"input-hint"},D={class:"attach-btn",title:"上传文件"},H=_({__name:"InputBar",props:c({disabled:{type:Boolean},placeholder:{}},{modelValue:{default:""},modelModifiers:{}}),emits:c(["send"],["update:modelValue"]),setup(s,{emit:u}){const t=b(s,"modelValue"),p=u,l=k();function h(e){e.key==="Enter"&&!e.shiftKey&&(e.preventDefault(),o())}function o(){const e=t.value.trim();e&&(p("send",e),t.value="")}function f(){const e=l.value;e&&(e.style.height="auto",e.style.height=e.scrollHeight+"px")}return(e,n)=>(V(),v("div",g,[a("div",w,[x(a("textarea",{ref_key:"textarea",ref:l,"onUpdate:modelValue":n[0]||(n[0]=m=>t.value=m),class:"input-textarea",placeholder:s.placeholder||"输入消息... (Enter 发送, Shift+Enter 换行)",disabled:s.disabled,rows:"1",onKeydown:h,onInput:f},null,40,z),[[y,t.value]]),a("button",{class:"send-btn",disabled:s.disabled||!t.value.trim(),onClick:o},[d(i(M),{size:18})],8,E)]),a("div",C,[a("button",D,[d(i(I),{size:14})])])]))}}),N=B(H,[["__scopeId","data-v-ea044f32"]]);export{N as I};
