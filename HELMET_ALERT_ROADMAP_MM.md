# Construction Safety Helmet Alert System — လမ်းညွှန်နှင့် Roadmap

ဒီဖိုင်က project ကို ဆက်လုပ်ရာမှာ မေ့သွားသည့်အချက်များ ပြန်ကြည့်နိုင်ရန် ရည်ရွယ်ထားသော မြန်မာဘာသာ လမ်းညွှန်ဖြစ်သည်။

---

## ၁။ Project ရည်ရွယ်ချက်

Webcam သို့မဟုတ် CCTV frame ထဲတွင် safety helmet မဝတ်ထားသောလူကို တွေ့လျှင် screen ပေါ်တွင် warning ပြပြီး alert sound ပေးမည့် system တည်ဆောက်ရန်။

လက်ရှိ requirement မှာ construction zone အတွင်း/အပြင် သို့မဟုတ် လူတစ်ယောက် site ထဲဝင်လာခြင်းကို မစစ်ပါ။ Webcam မှ detect လုပ်မိသော မည်သည့် `no_helmet` ကိုမဆို alert ပေးမည်။

### Model classes

```text
0 = helmet
1 = no_helmet
```

မူရင်း dataset ထဲက `head` class သည် helmet မဝတ်ထားသော bare head ကိုဆိုလိုသောကြောင့် YOLO dataset ပြောင်းချိန်တွင် `no_helmet` ဟု အမည်ပြောင်းထားသည်။

---

## ၂။ System ကို အပိုင်းနှစ်ပိုင်းခွဲနားလည်ရန်

### AI detection model

Frame ထဲမှ အရာဝတ္ထုကို အောက်ပါအတိုင်း detect လုပ်ပေးသည်။

```text
helmet
no_helmet
```

### Alert application

Model ကထုတ်ပေးသော detection result ကိုအသုံးပြုပြီး အောက်ပါ rule အတိုင်း alert ဆုံးဖြတ်သည်။

```text
no_helmet detected
+ confidence threshold ပြည့်
+ အချိန်အနည်းငယ် ဆက်တိုက်တွေ့
= alert
```

Model ကို alert ပေးရန် train လုပ်ခြင်းမဟုတ်ပါ။ Model က detection result ပေးပြီး Python application က alert ဆုံးဖြတ်ခြင်းဖြစ်သည်။

---

## ၃။ System flow

```text
Webcam frame
    ↓
YOLO helmet/no-helmet detection
    ↓
no_helmet ရှိ/မရှိ စစ်
    ↓
confidence threshold စစ်
    ↓
0.5–1 second ဆက်တိုက်တွေ့/မတွေ့ စစ်
    ↓
Warning box + alert sound + event log
    ↓
Cooldown သတ်မှတ်ပြီး ထပ်ခါထပ်ခါ alert မပေးအောင်ကာကွယ်
```

---

## ၄။ လက်ရှိ dataset အခြေအနေ

မူရင်း dataset:

```text
archive/
├── images/          5,000 images
└── annotations/     5,000 Pascal VOC XML files
```

စစ်ဆေးပြီးသောအချက်များ:

- Image နှင့် XML အရေအတွက် တူညီသည်။
- Image တိုင်းတွင် matching XML ရှိသည်။
- ပျက်နေသော XML မရှိပါ။
- Invalid bounding box မရှိပါ။
- Empty annotation မရှိပါ။

မူရင်း class distribution:

```text
helmet = 18,966 objects
head   =  5,785 objects
person =    751 objects
```

`person` annotation အလွန်နည်းပြီး လူအားလုံးကို ပြည့်စုံစွာ label လုပ်ထားခြင်းမဟုတ်နိုင်သောကြောင့် လက်ရှိ model မှာ `person` class ကိုမသုံးပါ။

YOLO အတွက်ပြောင်းပြီးသော dataset:

```text
construction_helmet_yolo/
├── data.yaml
├── images/
│   ├── train/       3,500 images
│   ├── val/         1,000 images
│   └── test/          500 images
└── labels/
    ├── train/       3,500 TXT labels
    ├── val/         1,000 TXT labels
    └── test/          500 TXT labels
```

Dataset split:

```text
Train      = 70%
Validation = 20%
Test       = 10%
```

---

## ၅။ Project roadmap checklist

### Phase A — Dataset ပြင်ဆင်ခြင်း

- [x] Construction helmet dataset ရွေးခြင်း
- [x] Image/XML pairs စစ်ခြင်း
- [x] XML bounding boxes စစ်ခြင်း
- [x] `head` ကို `no_helmet` အဖြစ်သတ်မှတ်ခြင်း
- [x] Pascal VOC XML မှ YOLO TXT သို့ပြောင်းခြင်း
- [x] Train/validation/test ခွဲခြင်း
- [x] `data.yaml` တည်ဆောက်ခြင်း
- [x] `helmet` နှင့် `no_helmet` နှစ် class သုံးရန်ဆုံးဖြတ်ခြင်း

### Phase B — Training environment

- [x] Python virtual environment ရှိခြင်း
- [x] Ultralytics install လုပ်ခြင်း
- [x] NVIDIA GTX 1650 4 GB ရှိကြောင်းစစ်ခြင်း
- [x] CUDA-enabled PyTorch install ပြီးစီးခြင်း
- [x] `torch.cuda.is_available()` သည် `True` ဖြစ်ကြောင်းစစ်ခြင်း

### Phase C — Model training

- [x] `train_helmet.py` တည်ဆောက်ခြင်း
- [x] One-epoch smoke test လုပ်ခြင်း
- [x] Generated training batch images တွင် labels မှန်ကြောင်းစစ်ခြင်း
- [ ] 30-epoch baseline experiment လုပ်ခြင်း
- [ ] Full training လုပ်ခြင်း
- [ ] Best model ကို test set ဖြင့် evaluate လုပ်ခြင်း

### Phase D — Webcam detection

- [ ] `best.pt` ဖြင့် image prediction စမ်းခြင်း
- [ ] Saved video ဖြင့် prediction စမ်းခြင်း
- [ ] Webcam live detection တည်ဆောက်ခြင်း
- [ ] Detection confidence threshold ရွေးခြင်း
- [ ] FPS တိုင်းတာခြင်း

### Phase E — Alert system

- [ ] `no_helmet` ဆက်တိုက်တွေ့သည့်အချိန်တွက်ခြင်း
- [ ] Alert message ပြခြင်း
- [ ] Alert sound ထည့်ခြင်း
- [ ] Alert cooldown ထည့်ခြင်း
- [ ] Screenshot နှင့် timestamp သိမ်းခြင်း
- [ ] False alert များကို field test လုပ်ခြင်း

### Phase F — နောက်ပိုင်းတိုးချဲ့နိုင်သည့်အရာများ

- [ ] Object tracking ထည့်ခြင်း
- [ ] လူတစ်ယောက်စီအလိုက် track ID သတ်မှတ်ခြင်း
- [ ] Track ID တစ်ခုစီအတွက် alert တစ်ကြိမ်သာပေးခြင်း
- [ ] Multiple webcams ထည့်ခြင်း
- [ ] Event database/dashboard ထည့်ခြင်း
- [ ] Email/Telegram/other notification ထည့်ခြင်း

---

## ၆။ CUDA PyTorch စစ်ဆေးခြင်း

GPU-enabled PyTorch တင်ပြီးနောက် PowerShell တွင်:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('Version:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

မျှော်လင့်ထားသော output:

```text
CUDA: True
GPU: NVIDIA GeForce GTX 1650
```

`CUDA: False` ဖြစ်နေလျှင် model training မစမီ PyTorch/CUDA installation ကို ပြန်စစ်ရန်။

---

## ၇။ Training file ကို run ခြင်း

Training program:

```text
train_helmet.py
```

Run command:

```powershell
.\.venv\Scripts\python.exe train_helmet.py
```

လက်ရှိ setting:

```python
SMOKE_TEST = True
```

ဒါကြောင့် ပထမ run မှာ epoch တစ်ခုပဲ train လုပ်မည်။ Dataset နှင့် training pipeline မှန်ကန်ကြောင်းစစ်ရန်ဖြစ်ပြီး accuracy ရရန်မဟုတ်ပါ။

Smoke test အောင်မြင်ပြီးမှ:

```python
SMOKE_TEST = False
```

ပြောင်းပြီး full training လုပ်ရန်။

---

## ၈။ Training parameters အဓိပ္ပာယ်

လက်ရှိ baseline settings:

```python
MODEL_NAME = "yolo26n.pt"
IMAGE_SIZE = 416
BATCH_SIZE = 8
DEVICE = 0
WORKERS = 0
FULL_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
RANDOM_SEED = 42
```

### `MODEL_NAME = "yolo26n.pt"`

`n` သည် Nano model ဖြစ်သည်။ GTX 1650 4 GB နှင့် real-time alert system အတွက် speed/memory အနေဖြင့် သင့်တော်သော baseline ဖြစ်သည်။ Accuracy မလုံလောက်မှ `s` model ကို နောက် experiment အဖြစ်စမ်းရန်။

### `IMAGE_SIZE = 416`

Dataset images များသည် 416×416 ဖြစ်သောကြောင့် baseline အဖြစ် 416 သုံးသည်။ Distant/small helmets မဖမ်းနိုင်လျှင် 640 ကို နောက် experiment အဖြစ်စမ်းနိုင်သော်လည်း GPU memory ပိုသုံးပြီး ပိုနှေးမည်။

### `BATCH_SIZE = 8`

GPU က တစ်ကြိမ်တွင် image ရှစ်ပုံ process လုပ်မည်။ CUDA out-of-memory ဖြစ်လျှင်:

```text
8 → 4 → 2
```

ဟုလျှော့ရန်။

### `FULL_EPOCHS = 100`

Dataset အားလုံးကို အများဆုံး အကြိမ် 100 လေ့လာမည်။ 100 ပြည့်အောင် မဖြစ်မနေ run မည်ဟု မဆိုလိုပါ။ Validation result မတိုးတော့လျှင် early stopping က စောစောရပ်နိုင်သည်။

### `EARLY_STOPPING_PATIENCE = 20`

Validation result မတိုးတက်ဘဲ epoch 20 ဆက်တိုက်ဖြစ်လျှင် training ရပ်မည်။ ဒီ patience က alert စောင့်ချိန်နှင့်မသက်ဆိုင်ပါ။

### `DEVICE = 0`

ပထမ NVIDIA GPU ကိုသုံးမည်။ CPU သုံးလျှင် `"cpu"` ဟုပြောင်းနိုင်သော်လည်း training အလွန်နှေးနိုင်သည်။

### `WORKERS = 0`

Windows မှာ စတင်စမ်းသပ်ရန် တည်ငြိမ်သော value ဖြစ်သည်။ အားလုံးအလုပ်လုပ်ပြီးမှ 2 သို့တိုးစမ်းနိုင်သည်။

### `RANDOM_SEED = 42`

Random operations များကို ပြန် run လျှင် အနီးစပ်ဆုံးတူညီသော experiment ဖြစ်စေရန်အသုံးပြုသည်။ 42 ကိုယ်တိုင်မှာ အထူး AI အဓိပ္ပာယ်မရှိပါ။

---

## ၉။ Smoke test ပြီးနောက် စစ်ရမည့်အရာများ

Result directory:

```text
runs/smoke_test/
```

အဓိကကြည့်ရန်:

```text
train_batch0.jpg
train_batch1.jpg
train_batch2.jpg
labels.jpg
results.csv
weights/best.pt
weights/last.pt
```

Batch preview ပုံများတွင်:

- Helmet ပေါ်တွင် `helmet` box ဖြစ်ရမည်။
- Bare head ပေါ်တွင် `no_helmet` box ဖြစ်ရမည်။
- Class နှစ်ခု ပြောင်းပြန်မဖြစ်ရ။
- Bounding box သည် object ကို မှန်ကန်စွာဝိုင်းထားရမည်။
- မဆိုင်သော object ကို box မဆွဲထားရ။

---

## ၁၀။ Experiment လုပ်နည်း

Parameter အများကြီးကို တစ်ပြိုင်နက်မပြောင်းရန်။ Experiment တစ်ခုမှာ အချက်တစ်ခုသာပြောင်းပြီး နှိုင်းယှဉ်ရန်။

ဥပမာ:

| Experiment | Model | Image size | Batch |
|---|---|---:|---:|
| Baseline | YOLO26n | 416 | 8 |
| Small-object test | YOLO26n | 640 | 4 |
| Model-size test | YOLO26s | 416 | 4 |

Experiment တစ်ခုစီအတွက် မှတ်တမ်းတင်ရန်:

```text
Experiment name:
Date:
Model:
Image size:
Batch size:
Epochs completed:
Best epoch:
Precision:
Recall:
mAP50:
mAP50-95:
No-helmet recall:
Webcam FPS:
Observed problems:
Decision:
```

---

## ၁၁။ Safety project အတွက် metrics ဦးစားပေးပုံ

အရေးကြီးဆုံးက `no_helmet recall` ဖြစ်သည်။

```text
No-helmet လူ 100 ယောက်ထဲမှ 92 ယောက်ဖမ်းမိ
→ Recall = 92%
```

### False negative

Helmet မဝတ်သူကို model က မဖမ်းမိခြင်း။ Safety project အတွက် ပိုအန္တရာယ်ကြီးသည်။

### False positive

Helmet ဝတ်ထားသူကို `no_helmet` ဟုမှား detect လုပ်ခြင်း။ Alert မှားပေးနိုင်သည်။

Final system ဆုံးဖြတ်ရာတွင် overall mAP တစ်ခုတည်းမကြည့်ဘဲ `no_helmet` class ၏ recall၊ precision နှင့် real webcam test ကိုကြည့်ရန်။

---

## ၁၂။ Webcam alert configuration အစပြုတန်ဖိုးများ

```python
NO_HELMET_CONFIDENCE = 0.35
CONFIRMATION_SECONDS = 0.75
ALERT_COOLDOWN_SECONDS = 10
```

### Confidence

နိမ့်လွန်းလျှင် false alerts များနိုင်ပြီး မြင့်လွန်းလျှင် no-helmet လူကိုလွတ်နိုင်သည်။ Test videos နှင့် webcam ပေါ်တွင် 0.25၊ 0.35၊ 0.50 တို့ကိုနှိုင်းယှဉ်ရန်။

### Confirmation time

Frame တစ်ခုမှား detect ဖြစ်ရုံနှင့် alert မပေးရန် `no_helmet` ကို 0.5–1 second ဆက်တိုက်တွေ့မှ အတည်ပြုသည်။

### Cooldown

Alert တစ်ခုပြီးနောက် frame တိုင်း ထပ်ခါထပ်ခါမမြည်ရန် 10 seconds ခန့်စောင့်သည်။ နောက်ပိုင်း tracker ထည့်လျှင် လူတစ်ယောက်စီအလိုက် cooldown ခွဲနိုင်သည်။

---

## ၁၃။ Alert logic pseudocode

```python
if no_helmet_detected and confidence >= 0.35:
    visible_time += frame_duration

    if visible_time >= 0.75 and cooldown_finished:
        show_red_warning()
        play_alert_sound()
        save_screenshot_and_time()
        start_cooldown()
else:
    visible_time = 0
```

ပထမ version မှာ webcam frame ထဲ `no_helmet` တစ်ခုခုရှိလျှင် global alert ပေးနိုင်သည်။ Version 2 မှာ tracking ထည့်ပြီး လူတစ်ယောက်စီအလိုက် alert ပေးနိုင်သည်။

---

## ၁၄။ Person–helmet association ကို အခုမလုပ်သေးသည့်အကြောင်း

အခြားနည်းလမ်းတစ်ခုမှာ person၊ helmet နှင့် head သုံးခုကို detect လုပ်ပြီး helmet က ဘယ် person ကိုပိုင်သည်ကို association လုပ်ခြင်းဖြစ်သည်။ ဒီနည်းသည် နောက်ပိုင်း advanced version အတွက်အသုံးဝင်သော်လည်း လက်ရှိမှာ:

- `person` annotation အလွန်နည်းသည်။
- Person အားလုံး label လုပ်ထားခြင်းမရှိနိုင်ပါ။
- Helmet detection လွတ်သွားလျှင် association logic က false alert ပေးနိုင်သည်။
- လူများနီးကပ်နေလျှင် helmet ကို person မှားပြီးတွဲနိုင်သည်။
- GTX 1650 ပေါ်တွင် additional processing ကြောင့် FPS ကျနိုင်သည်။

ဒါကြောင့် လက်ရှိ MVP မှာ `no_helmet` ကိုတိုက်ရိုက် detect လုပ်ပြီး alert ပေးခြင်းကိုသုံးသည်။

---

## ၁၅။ Common errors

### `CUDA: False`

CPU-only PyTorch install ဖြစ်နေခြင်း။ CUDA-enabled PyTorch ကို install လုပ်ပြီး ပြန်စစ်ရန်။

### `CUDA out of memory`

```text
BATCH_SIZE: 8 → 4 → 2
```

ဟုလျှော့ရန်။ လိုအပ်လျှင် image size ကိုလည်း လျှော့ရန်။

### `Dataset images not found`

`data.yaml` ထဲမှ train/val/test paths နှင့် dataset folder structure ကိုစစ်ရန်။ `path: .` ကို active setting အဖြစ်မထားရန်။

### Labels မပေါ်ခြင်း

Image filename နှင့် label filename stem တူ/မတူ စစ်ရန်။ ဥပမာ:

```text
hard_hat_workers10.png
hard_hat_workers10.txt
```

### Training ကောင်းပေမဲ့ webcam မှာမကောင်းခြင်း

Dataset domain နှင့် တကယ့် webcam environment မတူခြင်းဖြစ်နိုင်သည်။ ကိုယ်ပိုင် camera မှ images စု၊ label လုပ်ပြီး dataset ထဲထည့်ကာ fine-tune ပြန်လုပ်ရန်။

### Alert မကြာခဏမှားပေးခြင်း

- Confidence အနည်းငယ်တိုးရန်။
- Confirmation time တိုးရန်။
- Tracker/cooldown ထည့်ရန်။
- False-positive images ကို dataset ထဲထည့်ပြီး train ပြန်လုပ်ရန်။

---

## ၁၆။ Senior developer workflow

Project ကို ဒီအစဉ်အတိုင်းလုပ်ရန်:

```text
Dataset validity
→ Label visualization
→ One-epoch smoke test
→ Baseline training
→ Test-set evaluation
→ Saved-video inference
→ Webcam inference
→ Alert logic
→ Real-environment data collection
→ Fine-tuning
→ Deployment
```

Training ကိုအရင်ပြီးအောင်လုပ်ပြီးမှ code အားလုံးတစ်ခါတည်းဆောက်ရန်မလိုပါ။ အဆင့်တစ်ခုစီကို verify လုပ်ပြီးမှ နောက်တစ်ဆင့်သွားရန်။

---

## ၁၇။ Safety limitation

ဒီ system သည် university/learning prototype အဖြစ်အသုံးပြုရန်ဖြစ်သည်။ Real construction safety deployment မှာ AI alert တစ်ခုတည်းကို အားမကိုးသင့်ပါ။ Camera angle၊ lighting၊ occlusion၊ helmet color၊ image blur နှင့် model error များကြောင့် helmet မဝတ်သူကို လွတ်နိုင်သည်။ လူကြီးကြပ်မှုနှင့် တည်ဆဲ safety procedures များကို အစားမထိုးနိုင်ပါ။

---

## ၁၈။ အတိုချုံးမှတ်ထားရန်

```text
Dataset classes: helmet + no_helmet
Current model: YOLO26n
Current GPU: GTX 1650 4 GB
Baseline image size: 416
Baseline batch: 8
First run: 1 epoch smoke test
Full-training maximum: 100 epochs
Early stopping patience: 20 epochs
Primary safety metric: no_helmet recall
Alert: confidence + time confirmation + cooldown
Current design: no ROI, no person association
Future design: per-person tracking and event logging
```
