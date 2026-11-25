# التقرير النهائي للتجارب والمحاكاة

## إعداد التجربة

تم تنفيذ محاكاة Monte Carlo على صورة NASA/JPL-Caltech PIA24428 بعد تحويلها إلى grayscale بحجم 256×256. أصبحت الصورة payload ثابتًا من 524,288 bit، وتم استخدام row-major serialization وMSB-first داخل كل pixel.

تمت مقارنة ثلاث طرق:

- Uncoded transmission.
- Hamming (7,4) systematic code.
- LDPC بطول كتلة 1008 و\(k=506\)، مع log-domain sum-product belief propagation وحد أقصى 50 iterations.

نموذج القناة هو Binary Symmetric Channel مستقل البتات عند:

\[
p \in \{0, 0.001, 0.005, 0.01, 0.02, 0.05\}.
\]

تم استخدام trial واحد للتحقق عند \(p=0\)، و100 channel trials لكل طريقة عند كل قيمة nonzero من \(p\). تم تثبيت master seed يساوي 20261125، وتوليد child seed من:

```text
SeedSequence([master_seed, method_index, p_index, trial_index])
```

كل النتائج الموجودة في هذا التقرير ناتجة عن المحاكاة الفعلية، وليست قيمًا مفترضة.

## النتائج الرئيسية

| p | Method | Decoded BER | FER | Pooled PSNR (dB) | Mean SSIM | Mean decoder time (s) |
|---:|:---|---:|---:|---:|---:|---:|
| 0.001 | Uncoded | 9.9884e-4 | 1.00 | 34.72 | 0.9459 | 0.0048 |
| 0.001 | Hamming (7,4) | 9.2697e-6 | 0.93 | 55.13 | 0.9996 | 4.2537 |
| 0.001 | LDPC | 0 | 0 | ∞ | 1.0000 | 3.2090 |
| 0.005 | Uncoded | 4.9955e-3 | 1.00 | 27.74 | 0.7713 | 0.0034 |
| 0.005 | Hamming (7,4) | 2.2028e-4 | 1.00 | 41.63 | 0.9902 | 4.3653 |
| 0.005 | LDPC | 0 | 0 | ∞ | 1.0000 | 3.3990 |
| 0.010 | Uncoded | 1.0004e-2 | 1.00 | 24.75 | 0.6233 | 0.0034 |
| 0.010 | Hamming (7,4) | 8.7004e-4 | 1.00 | 35.65 | 0.9611 | 4.4708 |
| 0.010 | LDPC | 0 | 0 | ∞ | 1.0000 | 3.4986 |
| 0.020 | Uncoded | 1.9982e-2 | 1.00 | 21.75 | 0.4511 | 0.0033 |
| 0.020 | Hamming (7,4) | 3.3801e-3 | 1.00 | 29.74 | 0.8623 | 4.6405 |
| 0.020 | LDPC | 0 | 0 | ∞ | 1.0000 | 3.9056 |
| 0.050 | Uncoded | 4.9985e-2 | 1.00 | 17.79 | 0.2691 | 0.0033 |
| 0.050 | Hamming (7,4) | 1.9411e-2 | 1.00 | 22.17 | 0.5051 | 4.8116 |
| 0.050 | LDPC | 4.9343e-5 | 0.73 | 47.88 | 0.9981 | 4.9491 |

ملاحظة: قيم PSNR في هذا الجدول هي pooled-MSE PSNR، وليست المتوسط الحسابي لقيم PSNR الفردية. الرمز ∞ يعني أن كل trial في المجموعة أعاد الصورة exact reconstruction، أي إن متوسط MSE يساوي صفرًا.

## تفسير النتائج

### Uncoded transmission

الـ decoded BER في الحالة غير المشفرة قريب من قيمة \(p\) في كل مستويات الضوضاء، كما هو متوقع من BSC. ومع زيادة \(p\)، انخفضت جودة الصورة بوضوح؛ عند \(p=0.05\) أصبح mean SSIM حوالي 0.2691 وpooled PSNR حوالي 17.79 dB.

FER يساوي 1.0 تقريبًا عند جميع مستويات الضوضاء nonzero لأن الصورة تحتوي على 524,288 source bits. وجود خطأ واحد فقط يجعل الصورة trial فاشلًا وفق تعريف FER المستخدم. لذلك لا ينبغي تفسير FER وحده على أنه نسبة الصور غير القابلة للمشاهدة.

### Hamming (7,4)

قلل Hamming decoded BER بدرجة كبيرة مقارنةً بالإرسال غير المشفر. عند \(p=0.01\)، انخفض BER من حوالي \(1.0\times10^{-2}\) إلى \(8.7\times10^{-4}\)، وارتفع PSNR من 24.75 dB إلى 35.65 dB، كما ارتفع SSIM من 0.6233 إلى 0.9611.

مع ذلك، ظل FER مساويًا 1.0 عند معظم المستويات لأن Hamming (7,4) يصحح خطأً واحدًا فقط في كل block، بينما تحتوي الصورة على 131,072 Hamming blocks. حتى معدل residual صغير لكل block يؤدي غالبًا إلى وجود خطأ في مكان ما من الصورة. هذه النتيجة لا تعني أن الصورة كلها غير مفيدة بصريًا؛ لذلك يجب عرض BER وPSNR وSSIM معه.

### LDPC

في الإعداد التجريبي الحالي، أعاد LDPC الصورة exact في جميع التجارب عند \(p\leq0.02\)، ولذلك كان decoded BER وFER يساويان صفرًا وSSIM يساوي 1.0. عند \(p=0.05\)، انخفض decoded BER إلى \(4.9343\times10^{-5}\)، وبلغ FER 0.73، بينما ظل pooled PSNR حوالي 47.88 dB وmean SSIM حوالي 0.9981.

هذه النتيجة تعني أن LDPC المقترح كان أقوى من Hamming في configuration المحددة وتحت BSC المستخدم. لكنها لا تثبت أن كل LDPC code أفضل من كل Hamming code، لأن التجربة تستخدم code rates وأطوال blocks وميزانيات transmitted bits مختلفة.

## تكلفة الإرسال والتعقيد

الإعدادات ليست equal-rate:

- Uncoded: nominal rate = 1، effective payload rate = 1.
- Hamming: nominal rate = \(4/7\approx0.571429\)، effective payload rate = 0.571429.
- LDPC: nominal rate = \(506/1008\approx0.501984\)، effective payload rate ≈ 0.501569 بسبب 434 padding bits.

لذلك يحصل LDPC على redundancy أكبر قليلًا من Hamming، كما يرسل عددًا أكبر من coded bits. يجب أن تذكر الورقة أن هذه مقارنة بين complete tested configurations، وليست rate-matched code-family benchmark.

أزمنة decoder المقاسة هنا implementation-dependent. Hamming ظهر أبطأ من LDPC في هذه النسخة البرمجية لأن Hamming decoder منفذ بحلقة Python على 131,072 block، بينما LDPC decoder يستخدم عمليات vectorized. لا يجوز استنتاج أن Hamming أكثر تعقيدًا من LDPC عمومًا من هذه الأرقام وحدها.

## صياغة مقترحة لقسم Results

> The Monte Carlo experiment completed 100 independent channel trials for each nonzero BSC crossover probability and each transmission method. Uncoded transmission produced a decoded BER close to the channel crossover probability, whereas Hamming (7,4) reduced the residual bit error rate and improved both PSNR and SSIM. The selected LDPC configuration achieved exact image recovery in all trials up to \(p=0.02\). At \(p=0.05\), LDPC retained substantially lower decoded BER and higher image quality than both baselines, although its FER increased to 0.73. Because the three configurations use different nominal rates, block lengths, and transmitted-bit budgets, these results describe the tested configurations rather than an intrinsic universal ranking of the two code families.

## القيود التي يجب إبقاؤها في البحث

- تم اختبار صورة واحدة فقط.
- القناة BSC مستقلة البتات ولا تمثل burst errors أو fading أو synchronization loss أو packet loss.
- مصفوفة LDPC constructed educational matrix وليست CCSDS-compliant configuration.
- المقارنة ليست equal-rate ولا equal-energy.
- timing results تخص implementation وبيئة التشغيل الحالية.
- FER الصارم على مستوى الصورة حساس جدًا لأن الصورة تحتوي على عدد كبير من source bits.
- exact zero errors في 100 trial لا يثبت أن probability الحقيقية تساوي صفرًا؛ هو يثبت فقط عدم ظهور خطأ في العينة المختبرة. للـ FER عند صفر failures في 100 trial، يمكن الإبلاغ عن one-sided 95% upper bound يقارب 0.0295.
