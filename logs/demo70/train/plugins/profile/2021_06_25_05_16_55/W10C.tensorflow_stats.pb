"?9
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?z???@9?z???@A?z???@I?z???@a??kp????i??kp?????Unknown?
BHostIDLE"IDLE1??????@A??????@aܟܫ1R??i?q)?6???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1
ףp=k@9
ףp=k@A
ףp=k@I
ףp=k@aLT?vv?i{??c???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1u?V?h@9u?V?h@Au?V?h@Iu?V?h@a(Y1??t?i??
:????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1{?G??`@9{?G??`@A{?G??`@I{?G??`@a?{{^?k?irH??????Unknown
^HostGatherV2"GatherV2(1?Q??[M@9?Q??[M@A?Q??[M@I?Q??[M@ag???U_X?i?;VCD????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1y?&1|C@9y?&1|C@Ay?&1|C@Iy?&1|C@a??F?-P?iH_H?Z????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(15^?IrA@95^?IrA@A5^?IrA@I5^?IrA@a<tD?]?L?iep???????Unknown
o
HostSoftmax"sequential/dense_1/Softmax(1????xI>@9????xI>@A????xI>@I????xI>@a?L
q?$I?i????????Unknown
?HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??v???;@9??v???;@A??v???;@I??v???;@aӛ???G?iߩʩ????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1?O??n?7@9?O??n?7@A?O??n?7@I?O??n?7@aq4????C?i,??9?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1㥛? p6@9㥛? p6@A^?I?3@I^?I?3@a@?N?T@?i???;?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1}?5^??1@9}?5^??1@A}?5^??1@I}?5^??1@a?B??S=?i??͵M????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1Zd;?Om0@9Zd;?Om0@AZd;?Om0@IZd;?Om0@acʞqF;?i6c??????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1^?Ik-@9^?Ik-@A^?Ik-@I^?Ik-@a ?H1l8?i?c*
?????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1????xi(@9????xi(@A????xi(@I????xi(@a1?ED4?iG͒L????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?G?z?*@9?G?z?*@A??ʡ?&@I??ʡ?&@aA?Җ?2?iC????????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1??(\?B"@9??(\?B"@A??(\?B"@I??(\?B"@a????Q.?i????????Unknown
iHostWriteSummary"WriteSummary(1fffff?!@9fffff?!@Afffff?!@Ifffff?!@a ?K|JN-?i?|??\????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?n??? @9?n??? @A?n??? @I?n??? @a??!??+?i???????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1+??N@9+??N@A+??N@I+??N@a?2ps?T(?i
?J?????Unknown
`HostGatherV2"
GatherV2_1(1?rh??|@9?rh??|@A?rh??|@I?rh??|@a?/3?\?"?i=@???????Unknown
dHostDataset"Iterator::Model(1?"??~?5@9?"??~?5@A??~j??@I??~j??@aU????e!?i?\?????Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1?(\???@9?(\???@A?(\???@I?(\???@aQ?k?? ?i???}?????Unknown
ZHostArgMax"ArgMax(1?n???@9?n???@A?n???@I?n???@a? ?6bn ?i?*?c?????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?"??~?@9?"??~?@A?"??~?@I?"??~?@a??4L?i??????Unknown
[HostAddV2"Adam/add(1??/?$@9??/?$@A??/?$@I??/?$@a?????1?i6????????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1/?$??C@9/?$??C@A??Q??
@I??Q??
@a?04H?i?S??C????Unknown
lHostIteratorGetNext"IteratorGetNext(1u?V
@9u?V
@Au?V
@Iu?V
@a@$????i?"T??????Unknown
\HostArgMax"ArgMax_1(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a??h?8J?i?%'?????Unknown
V HostSum"Sum_2(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a??ڍ?)?iÔlv4????Unknown
?!HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1)\???(@9)\???(@A)\???(@I)\???(@a?J?e?i?妣?????Unknown
~"HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????x?@9????x?@A????x?@I????x?@aiX?7?0?i??8+Y????Unknown
e#Host
LogicalAnd"
LogicalAnd(1bX9??@9bX9??@AbX9??@IbX9??@a?(??i,)??????Unknown?
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1X9??v@9X9??v@AX9??v@IX9??v@a??4??i????q????Unknown
Y%HostPow"Adam/Pow(1????K7@9????K7@A????K7@I????K7@aؒ?????i[s?b?????Unknown
`&HostDivNoNan"
div_no_nan(1??(\??@9??(\??@A??(\??@I??(\??@a-?v%}?i6	?Vg????Unknown
?'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?Zd; @9?Zd;??A?Zd; @I?Zd;??a?????
?i???$?????Unknown
?(HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?G?z @9?G?z @A?G?z @I?G?z @a+l????
?inݻ?=????Unknown
[)HostPow"
Adam/Pow_1(1
ףp=
??9
ףp=
??A
ףp=
??I
ףp=
??a?IzS??	?iW+e?????Unknown
?*HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1?~j?t???9?~j?t???A?~j?t???I?~j?t???a?1?>??iH?]?????Unknown
o+HostReadVariableOp"Adam/ReadVariableOp(1333333??9333333??A333333??I333333??a??XV͔?i??<^????Unknown
u,HostReadVariableOp"div_no_nan/ReadVariableOp(19??v????99??v????A9??v????I9??v????ag?Ɩb?i?_??????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_4(1ףp=
???9ףp=
???Aףp=
???Iףp=
???a?s?s?i??)u????Unknown
X.HostEqual"Equal(1V-????9V-????AV-????IV-????a?t[???iRa??Z????Unknown
]/HostCast"Adam/Cast_1(1?A`??"??9?A`??"??A?A`??"??I?A`??"??a?????i~7??????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_3(1?|?5^???9?|?5^???A?|?5^???I?|?5^???a?C!?` ?i??Ĕ?????Unknown
X1HostCast"Cast_1(1??S㥛??9??S㥛??A??S㥛??I??S㥛??a?̇Y??>i?o?_ ????Unknown
t2HostReadVariableOp"Adam/Cast/ReadVariableOp(1?|?5^???9?|?5^???A?|?5^???I?|?5^???aF?-ro?>i???>[????Unknown
b3HostDivNoNan"div_no_nan_1(1??C?l??9??C?l??A??C?l??I??C?l??aD?0ӆ?>iW6?k?????Unknown
v4HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1/?$???9/?$???A/?$???I/?$???a?G"Z,?>i?ziĹ????Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1???x?&??9???x?&??A???x?&??I???x?&??a??5????>iQ?~??????Unknown
?6HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1??C?l???9??C?l???A??C?l???I??C?l???as`$N??>i??8????Unknown
{7HostSum"*categorical_crossentropy/weighted_loss/Sum(1B`??"???9B`??"???AB`??"???IB`??"???ace%?>i??1?/????Unknown
T8HostMul"Mul(1L7?A`???9L7?A`???AL7?A`???IL7?A`???a???%Y?>i:4R????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_1(1??C?l???9??C?l???A??C?l???I??C?l???af
??/??>i???@s????Unknown
v:HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?G?z???9?G?z???A?G?z???I?G?z???a?6L??V?>i?????????Unknown
y;HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1??? ?r??9??? ?r??A??? ?r??I??? ?r??a???^O?>i?!=?????Unknown
w<HostReadVariableOp"div_no_nan_1/ReadVariableOp(15^?I??95^?I??A5^?I??I5^?I??a?g}?k?>i~? ??????Unknown
a=HostIdentity"Identity(1H?z?G??9H?z?G??AH?z?G??IH?z?G??ag??׾??>iF?ߥ?????Unknown?
w>HostReadVariableOp"div_no_nan/ReadVariableOp_1(1^?I+??9^?I+??A^?I+??I^?I+??a??[?;?>i?????????Unknown
??HostDivNoNan",categorical_crossentropy/weighted_loss/value(1m???????9m???????Am???????Im???????a??\Z>?>i?????????Unknown*?8
uHostFlushSummaryWriter"FlushSummaryWriter(1?z???@9?z???@A?z???@I?z???@aP a?,??iP a?,???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1
ףp=k@9
ףp=k@A
ףp=k@I
ףp=k@a????9|w?i9????[???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1u?V?h@9u?V?h@Au?V?h@Iu?V?h@a?N??nzu?i??4?φ???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1{?G??`@9{?G??`@A{?G??`@I{?G??`@a?9sj?Bm?iB??????Unknown
^HostGatherV2"GatherV2(1?Q??[M@9?Q??[M@A?Q??[M@I?Q??[M@a?C&?{Y?i1թ?а???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1y?&1|C@9y?&1|C@Ay?&1|C@Iy?&1|C@a??R???P?i?~hE????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(15^?IrA@95^?IrA@A5^?IrA@I5^?IrA@a?!??aIN?i%hm??????Unknown
oHostSoftmax"sequential/dense_1/Softmax(1????xI>@9????xI>@A????xI>@I????xI>@ao?NKJJ?i??4Sj????Unknown
?	HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??v???;@9??v???;@A??v???;@I??v???;@a6????-H?i???u????Unknown
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1?O??n?7@9?O??n?7@A?O??n?7@I?O??n?7@a????vD?i8??B?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1㥛? p6@9㥛? p6@A^?I?3@I^?I?3@a?eΏ?A?i??/??????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1}?5^??1@9}?5^??1@A}?5^??1@I}?5^??1@a7?+??>?iQ_?(?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1Zd;?Om0@9Zd;?Om0@AZd;?Om0@IZd;?Om0@aS~????<?i?2ݿ=????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1^?Ik-@9^?Ik-@A^?Ik-@I^?Ik-@aK?}?/?9?iyB??n????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1????xi(@9????xi(@A????xi(@I????xi(@a?l-e?05?i'?b?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?G?z?*@9?G?z?*@A??ʡ?&@I??ʡ?&@a????ƌ3?i??<??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1??(\?B"@9??(\?B"@A??(\?B"@I??(\?B"@a?LX?_?/?i=v6́????Unknown
iHostWriteSummary"WriteSummary(1fffff?!@9fffff?!@Afffff?!@Ifffff?!@a?+9?D?.?i???l????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?n??? @9?n??? @A?n??? @I?n??? @a|n???&-?i?3??>????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1+??N@9+??N@A+??N@I+??N@aWS%?p)?i??}??????Unknown
`HostGatherV2"
GatherV2_1(1?rh??|@9?rh??|@A?rh??|@I?rh??|@a(,??8?#?i˘?????Unknown
dHostDataset"Iterator::Model(1?"??~?5@9?"??~?5@A??~j??@I??~j??@a?E|??0"?i????0????Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1?(\???@9?(\???@A?(\???@I?(\???@a?Ŝ?tS!?i[j?#F????Unknown
ZHostArgMax"ArgMax(1?n???@9?n???@A?n???@I?n???@a?-?.!?i,m?Y????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?"??~?@9?"??~?@A?"??~?@I?"??~?@a??\?^?iH?C????Unknown
[HostAddV2"Adam/add(1??/?$@9??/?$@A??/?$@I??/?$@aŝ͓LL?i??X????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1/?$??C@9/?$??C@A??Q??
@I??Q??
@a??v{3?i3?H??????Unknown
lHostIteratorGetNext"IteratorGetNext(1u?V
@9u?V
@Au?V
@Iu?V
@a#$N???i??s{????Unknown
\HostArgMax"ArgMax_1(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a,a???6?iO??*%????Unknown
VHostSum"Sum_2(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a???T6?i?7???????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1)\???(@9)\???(@A)\???(@I)\???(@aI?}U<?iW'N?g????Unknown
~ HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????x?@9????x?@A????x?@I????x?@a??4]9?i???????Unknown
e!Host
LogicalAnd"
LogicalAnd(1bX9??@9bX9??@AbX9??@IbX9??@as?????i_W'?????Unknown?
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1X9??v@9X9??v@AX9??v@IX9??v@a???t??i???B%????Unknown
Y#HostPow"Adam/Pow(1????K7@9????K7@A????K7@I????K7@a??d\ ??i?????????Unknown
`$HostDivNoNan"
div_no_nan(1??(\??@9??(\??@A??(\??@I??(\??@a????'??i ??&????Unknown
?%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?Zd; @9?Zd;??A?Zd; @I?Zd;??a?Gas.?i?????????Unknown
?&HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?G?z @9?G?z @A?G?z @I?G?z @a?ԝ???i??j????Unknown
['HostPow"
Adam/Pow_1(1
ףp=
??9
ףp=
??A
ףp=
??I
ףp=
??ap??}??
?iM?\1r????Unknown
?(HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1?~j?t???9?~j?t???A?~j?t???I?~j?t???av??i??i???i?????Unknown
o)HostReadVariableOp"Adam/ReadVariableOp(1333333??9333333??A333333??I333333??a??	O??i????3????Unknown
u*HostReadVariableOp"div_no_nan/ReadVariableOp(19??v????99??v????A9??v????I9??v????ar?A?O?i?*L?????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(1ףp=
???9ףp=
???Aףp=
???Iףp=
???a???n?ip???????Unknown
X,HostEqual"Equal(1V-????9V-????AV-????IV-????a?????i?<????Unknown
]-HostCast"Adam/Cast_1(1?A`??"??9?A`??"??A?A`??"??I?A`??"??aժ??X?i[c3g?????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_3(1?|?5^???9?|?5^???A?|?5^???I?|?5^???a??????i}????????Unknown
X/HostCast"Cast_1(1??S㥛??9??S㥛??A??S㥛??I??S㥛??a?nJ?' ?i?n??
????Unknown
t0HostReadVariableOp"Adam/Cast/ReadVariableOp(1?|?5^???9?|?5^???A?|?5^???I?|?5^???ap?/???>i{οH????Unknown
b1HostDivNoNan"div_no_nan_1(1??C?l??9??C?l??A??C?l??I??C?l??a??pU?F?>i\y??~????Unknown
v2HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1/?$???9/?$???A/?$???I/?$???a???m#?>icσ??????Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1???x?&??9???x?&??A???x?&??I???x?&??a?{:5??>iYD???????Unknown
?4HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1??C?l???9??C?l???A??C?l???I??C?l???a{??|???>iA>? ????Unknown
{5HostSum"*categorical_crossentropy/weighted_loss/Sum(1B`??"???9B`??"???AB`??"???IB`??"???a?e????>i?@&????Unknown
T6HostMul"Mul(1L7?A`???9L7?A`???AL7?A`???IL7?A`???a?s#?>i??'HJ????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1??C?l???9??C?l???A??C?l???I??C?l???a???	G?>i)?-?l????Unknown
v8HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?G?z???9?G?z???A?G?z???I?G?z???a:<?6<?>iC)? ?????Unknown
y9HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1??? ?r??9??? ?r??A??? ?r??I??? ?r??a?l7???>iz???????Unknown
w:HostReadVariableOp"div_no_nan_1/ReadVariableOp(15^?I??95^?I??A5^?I??I5^?I??a????q?>i??Y ?????Unknown
a;HostIdentity"Identity(1H?z?G??9H?z?G??AH?z?G??IH?z?G??aG??צ??>i[? ??????Unknown?
w<HostReadVariableOp"div_no_nan/ReadVariableOp_1(1^?I+??9^?I+??A^?I+??I^?I+??a&??4T?>i??T?????Unknown
?=HostDivNoNan",categorical_crossentropy/weighted_loss/value(1m???????9m???????Am???????Im???????a??A???>i?????????Unknown2Nvidia GPU (Kepler)