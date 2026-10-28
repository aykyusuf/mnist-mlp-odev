# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

Derin öğrenme dersi ödevi. MNIST'te sıfırdan bir MLP eğittim (CNN yok, hazır model yok), sonra iPad'de kendi el yazımla yazdığım okul numarasını (`23120205021`) bu modele tahmin ettirdim.

## Çalıştırma
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python mnist_mlp.py                                   # eğitim + deneyler -> outputs/
python predict_digits.py my_digits/numara.jpg --label 23120205021
python ablation_preprocess.py                         # önişleme adımlarını tek tek kapatma denemesi
```
Notebook'u (`mnist_mlp.ipynb`) açıp hücreleri sırayla çalıştırmak da aynı sonucu verir.

## Veri ve bölme
| Küme | Boyut | Kullanım |
|---|---|---|
| Train | 50000 | ağırlık güncelleme |
| Validation | 10000 | hiperparametre/model seçimi |
| Test | 10000 | **sadece en sonda**, baseline + validation'ın seçtiği model için bir kez |

MNIST'in 60k'lık eğitim setini seed 42 ile 50k/10k böldüm. 10k'lık test setine model seçerken hiç bakmadım. Görüntüler 784'e düzleştirildi ve `(x/255 - 0.1307) / 0.3081` ile normalize edildi.

## Deney 1 – Temel MLP
784 → 256 → ReLU → 128 → ReLU → 10, CrossEntropyLoss (softmax dahil), Adam (lr=1e-3), batch 128, 15 epoch, seed 42, CPU.

| Metrik | Değer |
|---|---|
| Training loss | 0.0087 |
| Validation loss | 0.0985 |
| Training accuracy | %99.70 |
| Validation accuracy | %97.84 |
| **Test accuracy** | **%97.92** |
| Eğitim süresi | 3.5 sn |

Eğitim eğrileri: ![baseline](outputs/curves_baseline.png)

## Hiperparametre karşılaştırması
Hepsi aynı seed, aynı bölme, 15 epoch, batch 128. Metrikler son epoch'tan. Test accuracy yalnızca baseline ve validation'a göre seçilen en iyi model için hesaplandı.

| config | hidden | optimizer | lr | momentum | dropout | train loss | val loss | train acc % | val acc % | test acc % | süre (sn) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 256-128 | adam | 0.001 | - | - | 0.0087 | 0.0985 | 99.70 | 97.84 | 97.92 | 3.5 |
| kucuk_agi | 64-32 | adam | 0.001 | - | - | 0.0246 | 0.1089 | 99.20 | 97.25 | - | 2.7 |
| buyuk_ag | 512-256-128 | adam | 0.001 | - | - | 0.0094 | 0.0982 | 99.68 | 97.94 | - | 7.9 |
| yuksek_lr | 256-128 | adam | 0.01 | - | - | 0.0757 | 0.2002 | 98.11 | 96.56 | - | 4.2 |
| sgd_momentum | 256-128 | sgd | 0.05 | 0.9 | - | 0.0002 | 0.0877 | 100.00 | 98.46 | 98.26 | 2.6 |
| dropout0.3 | 256-128 | adam | 0.001 | - | 0.3 | 0.0144 | 0.0719 | 99.56 | 97.90 | - | 4.2 |

![karşılaştırma](outputs/comparison_val.png)

Validation accuracy'ye göre en iyisi `sgd_momentum` çıktı.

### Sonuçlar hakkında
64-32'lik küçük ağ baseline'ın gerisinde kaldı (val %97.25, baseline %97.84), yani o kadar küçük olmamalı. Tersine, 512-256-128 ağı baseline'dan sadece %0.1 iyi ama iki katından uzun sürdü; büyütmenin faydası yok.

Adam'da lr'yi 1e-2'ye çıkarınca val loss 0.20'ye fırladı, accuracy %96.56'ya düştü. Adımlar çok büyük olduğu için minimumun etrafında zıplıyor olmalı.

SGD+momentum en iyi val accuracy'yi verdi ama train accuracy'si %100, train loss'u neredeyse sıfır. Train setini ezberlemiş. Train ile val arasındaki fark (~%1.5) baseline'dan küçük değil, o yüzden "en iyi" demek biraz fazla iddialı. Dropout ise accuracy'yi değiştirmedi, ama val loss'u en düşük yapan o oldu (0.072).

Genel olarak her config'te train acc val acc'den yüksek ve val loss train loss'un çok üstünde; 15 epoch'ta MLP train setini epey ezberliyor. Sadece tek seed ile çalıştım, bu yüzden config'ler arasındaki %0.1'lik farkları gürültü sayıyorum. Küçük ağ ve lr=1e-2 farkları ise yeterince büyük.

## Kendi el yazım
Numarayı iPad'de Apple Pencil ile, beyaz zemine siyah kalın kalemle yazdım (`my_digits/numara.jpg`).

Modele vermeden önce (`predict_digits.py`) şunları yapıyorum: gri tonlama, renk tersleme (MNIST'te zemin siyah, rakam beyaz; benim görselde tersi), rakamları ayırma, her rakamı en-boy oranını bozmadan 20×20'ye sığdırma, ağırlık merkezine göre 28×28'in ortasına koyma, MNIST'tekiyle aynı normalizasyon.

![önişlenmiş](outputs/preprocessed_numara.png)

İki model de 11 rakamın 11'ini doğru tahmin etti. Baseline'ın en az emin olduğu rakamlar ilk `2` (0.91) ve son `1` (0.86).

Rakamların ayrılması tam otomatik olmadı. `2-0`, `0-2` ve `0-2-1` birbirine değdiği için kod önce 8 parça buldu. Bu parçaları, 11 rakam olduğunu bilerek (`--label` uzunluğu) en az mürekkep olan yerden böldüm. Yani ayırma adımı rakam sayısını biliyor, hangi rakamlar olduğunu bilmiyor; tahmin bundan etkilenmiyor.

### MNIST test sonucu ile kendi el yazım arasındaki fark
MNIST testinde %97.9-98.3 aldım, kendi numaramda %100. Ama 11 rakamla "gerçek dünyada daha iyi" diyemem, sadece bu yazıda hata çıkmadı. Farkı daha çok güven değerlerinde görüyorum: SGD modeli her rakamda ~1.00, baseline ise `2` ve `1`'de 0.86-0.91'e düşüyor.

Başarının önişlemeden geldiğini `ablation_preprocess.py` ile denedim. Aynı modellere aynı rakamları adımları kapatarak verdim:

| Önişleme | baseline | sgd_momentum |
|---|---|---|
| Tam önişleme | 11/11 | 11/11 |
| Renk tersleme yok | 0/11 | 0/11 |
| 20×20 sığdırma + ortalama yok (kırpıp 28×28'e esnetme) | 1/11 | 2/11 |

Yani modeller rakamı değil, "MNIST gibi görünen" girdiyi tanıyor. Rakam MNIST'teki gibi ortalanıp ölçeklenmeyince model neredeyse hiç tutturamıyor. Bu, training verisi ile gerçek dünya verisi arasındaki farkın (distribution shift) en net görünen hali. Kağıda yazıp telefonla çekseydim (gölge, ince çizgi, eğik açı) büyük ihtimalle önişleme daha çok zorlanırdı; bunu denemedim.

## Dosyalar
- `mnist_mlp.ipynb` – aynı akış, çıktılar/grafikler notebook içinde (önce buna bak)
- `mnist_mlp.py` – veri, model, deneyler, grafikler (script sürümü)
- `predict_digits.py` – el yazısı önişleme + tahmin
- `ablation_preprocess.py` – önişleme adımlarını kapatma denemesi
- `my_digits/numara.jpg` – okul numaram
- `outputs/` – grafikler, `results.json`, loglar, model ağırlıkları
