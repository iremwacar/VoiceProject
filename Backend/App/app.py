from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import csv
import random
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import io
import base64
import matplotlib
matplotlib.use('Agg')
import joblib
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.io import wavfile
import librosa.display
from flask_socketio import SocketIO, emit
import noisereduce as nr
import threading
import sounddevice as sd
import wave
import whisper
import re
from collections import defaultdict
from zemberek import TurkishMorphology
# Zemberek çözümleyici


morphology = TurkishMorphology.create_with_defaults()

app = Flask(__name__, template_folder='../../Frontend')
socketio = SocketIO(app, cors_allowed_origins="*")
model = joblib.load('../Model/model.pkl')
whisper_model = whisper.load_model("base")

UPLOAD_FOLDER = '../Audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
    
# Kategori anahtar kelimeleri
CATEGORY_KEYWORDS = {
    "Spor": [
        "futbol", "basketbol", "voleybol", "tenis", "kriket", "golf", "maraton", 
        "spor", "şampiyona", "fitness", "antrenman", "koşu", "maç", "hakem", 
        "puan", "gol", "turnuva", "antrenör", "stadyum", "rakip", "kulüp"
    ],
    "Sağlık": [
        "hastane", "doktor", "tedavi", "ilaç", "diyet", "beslenme", "egzersiz", 
        "vitamin", "aşı", "sağlık", "terapi", "psikoloji", "hastalık", "kalp", 
        "grip", "bağışıklık", "kanser", "ameliyat", "tansiyon", "nabız", 
        "check-up", "laboratuvar", "tahlil"
    ],
    "Teknoloji": [
        "bilgisayar", "telefon", "yapay zeka", "robot", "internet", "donanım", 
        "kodlama", "yazılım", "mobil", "blockchain", "kripto para", "drone", 
        "veri", "otomasyon", "5G", "IoT", "elektronik", "bulut bilişim", 
        "siber güvenlik", "bilişim", "big data", "robotik", "teknoloji"
    ],
    "Sanat": [
        "resim", "heykel", "fotoğrafçılık", "edebiyat", "şiir", "roman", 
        "hikaye", "film", "tiyatro", "bale", "opera", "müzik", "keman", 
        "piyano", "gitar", "davul", "performans", "sergi", "galeri", "sanatçı"
    ],
    "Ekonomi": [
        "borsa", "yatırım", "kripto para", "döviz", "fiyat", "enflasyon", 
        "tasarruf", "işsizlik", "maliye", "kredi", "gelir", "harcama", 
        "ekonomi", "faiz", "piyasa", "ticaret", "vergi", "bütçe", "finans", 
        "borsa endeksi", "iş dünyası", "kazanma"
    ],
    "Eğitim": [
        "okul", "öğretmen", "öğrenci", "ders", "üniversite", "kitap", "sınav", 
        "araştırma", "ödev", "çalışma", "müfredat", "kurs", "sertifika", 
        "matematik", "tarih", "fen bilimleri", "edebiyat", "akademi", "not", 
        "başarı", "öğretim", "eğitim sistemi"
    ],
    "Dünyadan Haberler": [
        "haber", "dünya", "politik", "savaş", "barış", "diplomasi", "seçim", 
        "protesto", "uluslararası", "anlaşma", "ekonomi", "lider", "gündem", 
        "baskı", "kriz", "doğal afet", "küresel", "mülteci", "terör", "birleşmiş milletler"
    ],
    "Tarih": [
        "imparatorluk", "antikkent", "medeniyet", "arkeoloji", "müze", "kazı", 
        "krallar", "hanedan", "savaş", "barış antlaşması", "zafer", "devrim", 
        "antik dönem", "osmanlı", "roma", "moğollar", "ortaçağ", "ilk çağ", 
        "yeni çağ", "çağdaş tarih", "destan", "yazıt", "kalıntı", "efsane"
    ],
    "Çocuklar": [
        "çocuk", "bebek", "oyun", "oyuncak", "eğitim", "masal", "hikaye", 
        "çizgi film", "aktivite", "park", "kreş", "anaokulu", "beslenme", 
        "çocuk şarkısı", "çizgi film karakteri", "aile"
    ],
    "Hava Durumu": [
        "hava", "yağmur", "güneşli", "fırtına", "kar", "bulutlu", "sis", 
        "soğuk", "sıcaklık", "rüzgar", "nem", "mevsim", "ilkbahar", "yaz", 
        "sonbahar", "kış", "hava tahmini", "meteoroloji", "iklim", "dolu", 
        "hortum"
    ],
    "Bilim": [
        "araştırma", "buluş", "keşif", "laboratuvar", "deney", "biyoloji", 
        "fizik", "kimya", "astronomi", "genetik", "tıp", "teknoloji", 
        "evrim", "mikroskop", "bilimsel çalışma", "uzay", "gezegen", 
        "nasa", "kuantum", "çevre bilimi", "enerji", "atom", "parçacık fiziği"
    ],
    "Oyun": [
        "oyun", "video oyunu", "playstation", "xbox", "bilgisayar oyunu", 
        "mobil oyun", "şampiyonluk", "oyuncu", "espor", "fps", "rpg", 
        "minecraft", "fortnite", "valorant", "league of legends", 
        "counter strike", "şifreleme", "simülasyon", "macera", "bulmaca"
    ],
    "Sosyal Hayat": [
        "arkadaş", "aile", "parti", "gezi", "tatil", "toplantı", "organizasyon", 
        "iletişim", "sosyal medya", "sinema", "alışveriş", "buluşma", "etkinlik", 
        "düğün", "yemek", "sosyal sorumluluk", "eğlence", "paylaşım"
    ],
    "Yemek": [
        "yemek", "tarif", "mutfak", "kahvaltı", "tatlı", "restoran", 
        "fast food", "salata", "çorba", "pizza", "makarna", "ızgara", 
        "vegan", "sebze", "meyve", "baharat", "lezzet", "menü", "şef", "dondurma"
    ],
    "Genel Sohbet": [
        "merhaba", "günaydın", "görüşürüz", "nasılsın", "bugün", "evet", 
        "hayır", "peki", "belki", "olabilir", "konu", "anlamadım", "şaka", 
        "güzel", "keyif", "zaman", "bazen", "aslında", "düşünce", "hayat"
    ],
    "Bilim ve Teknolojik İlerlemeler": [
        "yapay zeka", "robot", "uzay keşfi", "mars", "ay", "nasa", "rover", 
        "kuantum bilgisayar", "genetik mühendisliği", "crispr", "biyoteknoloji", 
        "nanoteknoloji", "elektrikli araç", "dronelar", "güneş enerjisi", 
        "hidrojen enerjisi", "uzay teleskobu", "grafen", "moleküler biyoloji", 
        "robotik cerrahi", "füzyon enerjisi", "yeni materyaller", 
        "büyük hadron çarpıştırıcısı", "parçacık fiziği", "yapay organlar", 
        "çevre bilimi", "iklim teknolojisi"
    ],
     "Çevre ve Doğa": [
        "iklim değişikliği", "çevre", "orman", "biyoçeşitlilik", "yenilenebilir enerji",
        "geri dönüşüm", "çevre koruma", "karbon salınımı", "doğal afet", "orman yangını",
        "sel", "doğa yürüyüşü", "hayvanlar", "deniz", "orman yaşamı", "sürdürülebilirlik",
        "çevre bilinci", "plastik atıklar"
    ],
    "Uzay Bilimi ve Astronomi": [
        "gezegenler", "yıldız", "galaksi", "uzay teleskobu", "kara delik", 
        "nötron yıldızı", "evrenin genişlemesi", "uzay istasyonu", 
        "mars görevleri", "ay keşfi", "exoplanet", "astronot", 
        "rover", "uzay aracı", "güneş sistemi", "komet", "asteroid", 
        "uzay yarışı", "supernova", "big bang"
    ],
    "Genetik ve Biyoteknoloji": [
        "dna", "genetik kod", "crispr", "gen terapisi", "genom düzenleme", 
        "biyoteknoloji", "hücre mühendisliği", "moleküler biyoloji", 
        "insan genom projesi", "yeni ilaçlar", "bakteri genetiği", 
        "genetik modifikasyon", "bitki genetiği", "klonlama", "yapay doku", 
        "hücre yenilenmesi", "hastalık genetiği", "gen aktarımı", 
        "genom analiz", "epigenetik"
    ],
     "Doğa Olayları": [
        "yağmur", "fırtına", "sel", "deprem", "volkan", "dolu", "tsunami",
        "kar", "çığ", "orman yangını", "gök gürültüsü", "kasırga", "hortum",
        "şimşek", "tornado", "kuraklık", "meteoroloji", "doğal afet", "iklim değişikliği"
    ],
    "Enerji ve Çevre Teknolojileri": [
        "güneş enerjisi", "hidrojen enerjisi", "rüzgar enerjisi", 
        "yenilenebilir enerji", "iklim değişikliği", "karbon salınımı", 
        "sıfır emisyon", "karbon yakalama", "sürdürülebilirlik", 
        "biyoyakıtlar", "nükleer enerji", "enerji verimliliği", 
        "çevre bilimi", "plastik geri dönüşüm", "orman koruma", 
        "su arıtma", "atık yönetimi", "iklim çözümleri", 
        "elektrikli araç", "akıllı şehir"
    ],
      "Memleket": [
        "köy", "kasaba", "şehir", "memleket", "sokak", "komşular", "köy kahvesi",
        "çiftçilik", "tarla", "bağ", "bahçe", "yerel halk", "anılar", 
        "memleket yemekleri", "mahalli", "folklor", "yerel müzik", "memleket havası", "bölge kültürü"
    ],
     "Ulaşım ve Seyahat": [
        "uçak", "tren", "otobüs", "tatil", "seyahat planları", "yolculuk", 
        "bilet", "otogar", "havalimanı", "turizm", "yurt dışı", "vize", "oteller", 
        "kamping", "doğa gezileri", "şehir turu", "seyahat rehberi", "tatil köyü"
    ]
}    

emotions = {
    "Mutluluk": [
    "mutlu", "neşeli", "sevinçli", "coşkulu", "keyifli", "şen", "güleryüzlü", "rahat",
    "huzurlu", "sevgi", "umut", "memnun", "tatmin", "iyimser", "heyecanlı", "şanslı", 
    "güvenli", "güvende", "bahtiyar", "sevecen", "zevkli", "hoş", "gülmek", "eğlenceli", 
    "memnuniyet", "ferah", "mutluluk", "kahkaha", "şükür", "şükret", "gönül rahatlığı", 
    "sevinç", "şenlik", "kutlama", "neşe", "gönül ferahlığı", "hoşnut", "tatlı", "şefkatli", 
    "kalpten", "başarılı", "doyum", "tutkulu", "eğlen", "bayram", "bayramlık", "ilham", 
    "sevindir", "tatmin ol", "zafer", "yaşasın", "doğru", "güzel", "yükselmek", "coşku", 
    "neşelendirmek", "neşelendirici", "gülüş", "keyif", "mutlu olmak", "iç huzuru", "sevinçli", 
    "gönül alıcı", "mutluluk verici", "gönül rahatlatıcı", "neşelilik", "şeker gibi", 
    "güldürmek", "büyük mutluluk", "mutlu etmek", "şanslı olmak", "sevinçle dolu", "neşelenmek", 
    "iyi hissetmek", "neşeyle", "motive olmak", "yaşam sevinci", "güzel anlar", "kahkahalar", 
    "yaşamda umut", "sağlık", "güvenli hissetmek", "iyi durumda olmak", "başarı hissi", 
    "daha iyi", "memnuniyetle", "mutluluk kaynağı", "gözler gülmekte", "rahatlamak", "çağrılmak", 
    "tatmin olmuş", "hayatın tadını çıkarmak", "şanslı hissetmek", "eğlenceli anlar", "neşe kaynağı", 
    "kalp dolusu", "coşkulu anlar", "gönül okşamak", "moral", "huzurlu hissetmek", "keyif almak", 
    "güzel bir gün", "sevinçli anlar", "gülerken", "hoşnutluk", "mutluluk içinde", "zevk almak", 
    "şanslı olmak", "yükselmiş hissetmek", "tutkulu olmak", "kendisini iyi hissetmek", "eğlenmek"
],

"Üzüntü": [
    "üzgün", "hüzünlü", "kederli", "ağla", "yas", "elem", "acı", "dert", "kırgın", 
    "düşkün", "kahır", "üzüntü", "pişman", "melankolik", "kıskanç", "mahzun", "hüzün", 
    "bunalım", "üz", "üzül", "kırılgan", "dertli", "küs", "boşluk", "yoksun", 
    "mağdur", "umutsuz", "kayıp", "yalnız", "hiçlik", "çök", "bıkkın", "mutsuz", "hasret", 
    "gözyaşı", "ağlayış", "özlem", "mızmız", "yorgunluk", "karamsar", "karanlık", "sıkıntı", 
    "bıkkınlık", "dert et", "melankoli", "sızlan", "üzücü", "kırıl", "kırılmış", "hayal kırıklığı",
    "kötü hissetmek", "kötü", "hasta ol", "hasta", "hüzünlü olmak", "ruhsuz", "yorgun", 
    "umutsuzluk", "solgun", "gergin", "iç kararmış", "bunalmak", "yıkık", "huzursuz", "çöküş", 
    "bozulmuş", "derin üzülme", "iç sıkıntısı", "yaralı", "düşüş", "duygusal boşluk", 
    "hüzün içinde", "uzun süre üzülmek", "ağlamak", "gözyaşı dökmek", "umutsuzca", "bunalmak", 
    "sevgisizlik", "mutsuzluk", "endişe", "düş kırıklığı", "yıkılmış", "çaresiz", "çaresizlik", 
    "hiçbir şeyin önemi yok", "acı içinde", "canı sıkılmak", "kayıp duygusu", "yaklaşmayan umut", 
    "bulanık ruh hali", "depresyon", "sıkıntılı", "can sıkıntısı", "uzun süre yalnız", "kötü durum", 
    "acıklı", "üzüntü içinde", "ruhsuzluk", "baskı", "umutsuzca beklemek", "başarı eksikliği", 
    "içsel boşluk", "dönemsel bunalım", "devam eden üzüntü", "yokluk hissi", "derin düşünceler", 
    "kapalı hissetmek", "kırgınlık", "kötü hissetmek", "işlerin kötü gitmesi", "çözümsüzlük"
],
"İğrenti": [
    "iğrenç", "tiksinti", "midem bulanıyor", "nefret", "horgörü", "hoşlanma", "itici", 
    "mide bulantısı", "yadırga", "sevimsiz", "sıkıntılı", "bıkkınlık", "huysuz", "doyumsuz", 
    "hoşnutsuz", "tiksinmek", "pis", "çirkin", "sevimsizlik", "kaba", 
    "acımasız", "kibir", "yabancı", "korkutucu", "tehdit", "ürkütücü", "berbat", "sıkıntı", 
    "sorumsuz", "şikayet", "istenmeyen", "rahatsızlık", "düzensiz", "uygunsuz", "şaibeli", 
    "yakışıksız", "çirkinlik", "sevimsiz bir hal", "tatsızlık", "yadsımak", "menfi", 
    "hoşgörüsüzlük", "beğenmeme", "negatif", "suçla", "rahatsız et", "kötü niyetli", "saldırgan", 
    "iğrenç bir koku", "çirkinlik", "korkunç", "çirkinleşmiş", "itici bir ses", "çirkinleşmek", 
    "berbat durum", "rahatsız edici", "sıkıcı", "kapalı", "iğrenç bir tat", "berbat bir durum", 
    "tiksindirici", "sevilmeyen", "acımasızca", "rahatsız edici bir ortam", "şiddetli nefret", 
    "rahatlık bozucu", "can sıkıcı", "sinir bozucu", "tatsız", "hoş olmayan", "kaba davranış", 
    "hoşnutsuzluk", "öfkeli", "hoş olmayan his", "negatif duygular", "sinir bozucu", "karamsar", 
    "tiksindirici bir tavır", "kötü niyetli tavır", "kırıcı", "can sıkıcı", "tutarsız", "iğrenç bir insan", 
    "acımasızca bakmak", "hoş olmayan", "şiddetli tepki", "tutarsızlık", "güvensizlik", "hoş olmayan şey", 
    "düşmanlık", "hoşlanmamak", "rahatsız edici bir görüntü", "kötü niyetle yaklaşmak", "negatif bir etki", 
    "güvensiz bir durum", "gerilim", "tiksindirici bir davranış", "olumsuz", "rahatsızlık yaratmak", 
    "duygusal bozukluk", "kirli", "hoşlanılmayan bir tutum", "rahatsız edici", "sıkıntı yaratıcı", 
    "görülmesi istenmeyen", "iğrenç bir bakış", "hoş olmayan bir koku", "şüpheli", "hoşlanmadık", "kaba bir davranış"
],
"Korku": [
    "kork", "korku", "ürkmek", "ürkek", "tedirgin", "tehdit", "tehlikeli", "şüpheli", 
    "korkunç", "endişe", "ürkütücü", "panik", "dehşet", "şok", "ürkme", "kaçmak", 
    "saklanmak", "güvensizlik", "korkutucu", "ürkütmek", "panik olmak", "üzüntülü", "saldırı", 
    "korkmuş", "kaygı", "korkutan", "ürkütme", "korkulu", "ürkeklik", "tehlike", "kaçış", 
    "ürkek tavır", "gerginlik", "şüphecilik", "telaş", "çaresizlik", "şiddet", "şiddetli", 
    "gerilmek", "gerilmek", "gerilim", "terör", "ürkütmek", "ürkek", "tutsaklık", "tehlike", 
    "yalnızlık", "çaresiz", "dehşet içinde", "panik atak", "tehditkar", "karanlık", "kabus", 
    "tereddüt", "sıkıntı", "sinir", "yavaşlamak", "tekrar etmek", "güvensiz", "düşme korkusu", 
    "kaçmak zorunda", "baskı", "panik içinde", "fobi", "görüşsüz", "gerilim içinde", "korkunç bir ses", 
    "görünmeyen tehlike", "korku duymak", "aniden", "gerçek dışı", "şüpheli durum", "gizli tehlike", 
    "korkudan titremek", "kaotik", "umutsuzluk", "korku içinde olmak", "kaos", "belirsizlik", 
    "baskı altında", "korku ve endişe", "insanlık dışı", "korkutulmak", "düşman", "panikle hareket etmek", 
    "tartışma", "korkunç durum", "korku yaratmak", "düşmanlık", "garip", "belirsizlik", "kaybolmak", 
    "gergin ortam", "esrarengiz", "karanlıkta kalmak", "stres", "bağırmak", "güvensizlik duygusu", 
    "kapanmak", "yakın tehlike", "garip bir his", "savaş", "şiddetli korku", "savaşmaya karar vermek", 
    "kapanmak", "korkuyla yüzleşmek", "belirsiz olmak", "öldürülme korkusu", "yükselmek", "savaşmaya başlamak"
],
"Pişmanlık": [
    "pişman", "keşke", "hata", "yanılgı", "üzgün", "af dilemek", "özür", "yanlış", 
    "telafi", "mahcup", "sıkıntı", "üzüntü", "kayıp", "yanlışlık", "yanılsama", "utanç", 
    "özlem", "eksiklik", "kırgınlık", "bilmeyerek", "üzmek", "üzülmek", "çelişki", "karmaşa", 
    "karışıklık", "kendi kendine kızmak", "kendiyle çelişmek", "geç kalmak", "kaybetmek", 
    "kayıp", "mahcup olmak", "yanılmak", "hatırlamak", "kendi hatası", "sıkıntı yaşamak", 
    "geriye dönmek", "geri istemek", "affetmek", "affedilmek", "çıkışsız", "düşünceli", 
    "karışıklık", "şaşkınlık", "zaman kaybı", "kendiyle mücadele", "yetersizlik", "vicdan azabı", 
    "kayıtsızlık", "vicdan", "suçlu hissetmek", "keşke yapmasaydım", "kendi hatasıyla yüzleşmek", 
    "geri almak", "hatalı olmak", "suçluluk", "dönemek", "yakalamak", "pişmanlık duymak", 
    "geçmişi düşünmek", "tartışmak", "kendi kendine konuşmak", "söylediklerine pişman olmak", 
    "gerçekle yüzleşmek", "hatalı olduğunu fark etmek", "kayıpları anlamak", "pişmanlıkla dolu olmak", 
    "gizli suçluluk", "dönememek", "yanılmadığını anlamak", "hatalı olma hissi", "özür dilemek", 
    "anlatmak", "affedilmek için mücadele", "geri almak", "yapılacak şeyleri düşünmek", 
    "kendini suçlamak", "geri dönmek", "vazgeçmek", "keşke bir şansım olsaydı", "geçmişi değiştirmek", 
    "suçluluk duygusu", "tutarsızlık", "yapılacak şeyler", "geri adım atmak", "gizliden suçlu hissetmek", 
    "zamanı kaybetmek", "belirsizlik içinde", "gözyaşı dökmek", "görüşsüz olmak", "geriye bakmak", 
    "pişmanlıkla yüzleşmek", "duygusal karmaşa", "akıldan çıkmamak", "affedilme isteği", "kendisini suçlamak", 
    "suçlu olmak", "yeniden başlamayı dilemek", "olmazsa olmaz" 
],
"Şaşkınlık": [
    "şaşkın", "şaşırmak", "hayret", "şaşırmış", "şaşkınlık", "anlamamak", "hayret etmek", 
    "şok olmak", "inanamamak", "hayal kırıklığı", "hayranlık", "şaşılacak", "şaşmak", 
    "şüphelenmek", "anlayamamak", "merak etmek", "ilginç", "şaşkınca", "beklenmedik", 
    "şaşırtıcı", "şaşarak", "şaşmış", "hayretler içinde", "afallamak", "afallamış", 
    "düşünmek", "şaşkın halde", "şaşıp kalmak", "şaşkın kalmak", "etkilenmek", "şaşırtmak", 
    "ters köşe", "ters", "şaşılası", "şaşırtıcı olay", "şaşırtıcı durum", "şaşkınlığa uğramak", 
    "şaşakalmak", "şaşırıp kalmak", "şaşırmakta", "şaşkın bir şekilde", "hayran kalmak", 
    "şaşırtıcı bir durum", "şaşkınlığını gizleyememek", "şaşkınlıkla izlemek", "şaşkınca bakmak", 
    "hayran", "şaşkın gözlerle", "şaşmakta", "şaşkın gözle", "hayret içinde", "gözlerine inanmak", 
    "gözyaşlarıyla şaşırmak", "gözlerini açmak", "şüpheyi anlamak", "yeni bir keşif", "bir şeyler yanlış gitmek", 
    "belirsizlik", "şaşkın şekilde", "duygusal karmaşa", "karışıklık", "yenilik", "düşünmeden tepki vermek", 
    "akıl almaz", "tereddüt", "şok edici", "anlam verememek", "şaşkınlıkla bakmak", "sürpriz", "gözlerini ovuşturmak", 
    "şaşkınlıkla cevaplamak", "düşünmekte zorlanmak", "şok edici durum", "yeni bir bakış açısı", 
    "beklenmedik bir gelişme", "içinden çıkamamak", "dehşet", "akıl karışıklığı", "hiç beklenmeyen", 
    "sürpriz bir an", "şüpheye düşmek", "anı anlamamak", "zihnini zorlamak", "göz kamaştırıcı", 
    "yeni bir anlayış", "çaresiz bir bakış", "belirsizlik içinde kalmak", "kaybolmuş hissetmek", "adeta şaşkına dönmek"
],
"Sinirli": [
    "sinirli", "öfke", "sinir", "kızgın", "kızmak", "bağırmak", "öfkeyle", "kırgın", 
    "çılgın", "gergin", "tepkili", "sinirlenmek", "öfke patlaması", "sabırsız", "agresif", 
    "çatışma", "tartışma", "kavgacı", "şiddet", "kaba", "tutarsız", "öfke dolu", "sinir bozucu", 
    "hırçın", "asabi", "huzursuz", "öfkelenmek", "öfkeyi bastırmak", "sıkıntı", "öfkesini kontrol edememek", 
    "kontrolsüz", "kavga", "kaba davranış", "tepki göstermek", "sabırsızlık", "öfke duymak", 
    "öfke ile bağırmak", "şiddetle bağırmak", "nefretle", "hırs", "kıskançlık", "hiddet", 
    "öfkesini gizlemek", "öfkeyle konuşmak", "hırçınlık", "kırıcı", "öfkesini açığa vurmak", 
    "sabırsızlanmak", "öfkeli bir şekilde", "gerginlik", "sert", "kaba sözler", "patlamış sinir", 
    "gerilim", "tehdit", "kontrolsüz davranış", "duygusal patlama", "kızgınlıkla", "kavga etmek", 
    "öfke dolu bakış", "yıkıcı", "öfke nöbeti", "sözlü saldırı", "huzursuzluk", "sinir krizi", 
    "delirtilmiş", "nefsini kaybetmek", "tartışmak", "sinir bozucu durum", "yıkıcı davranış", 
    "ağır konuşmalar", "öfkeli gözler", "nefretle konuşmak", "geri tepmek", "geri basmak", "sabırsız davranmak", 
    "sinirli tavırlar", "öfke içinde", "sözcüklerle saldırmak", "düşmanlık", "sinirle", "öfkesini dışa vurmak", 
    "öfke duygusu", "ağırbaşlısızlık", "kavga çıkarmak", "öfke izleri", "nefret duygusu", "sinirli bir şekilde", 
    "soğukkanlılık kaybolmuş", "öfke dolu davranış", "saldırganlık", "sabırsızlık duygusu", "asabi bakış"
],
"Merak": [
    "merak", "soru", "şüphe", "belirsizlik", "ilgi", "takip etmek", "araştırmak", 
    "kafasında soru işareti", "hayal etmek", "meraklı", "bilgi arayışı", "ilginç", 
    "incelemek", "gözlem", "heyecan", "merak etmek", "sürükleyici", "çözmek", 
    "gizem", "merak uyandırıcı", "keşfetmek", "meraklı bir şekilde", "öğrenmek", 
    "bilmek istemek", "anlamaya çalışmak", "sır", "merak duygusu", "merak içinde", 
    "meraklı bakış", "merak uyandıran", "merakla", "heyecanlı bekleyiş", "bilgi edinmek", 
    "merakla izlemek", "merak içinde olmak", "merakla soru sormak", "araştırma yapmak", 
    "keşif yapmak", "bilgiyi açığa çıkarmak", "meraklı bir tavır", "sırlı", "merak edilen", 
    "çözülmemiş", "merak duygusu uyandıran", "soru sormak", "merakla öğrenmek", 
    "merakla gözlemek", "keşfetmek için sabırsızlanmak", "anlatılmamış", "farkında olmak", 
    "yeni şeyler öğrenmek", "merakla araştırmak", "heyecanla beklemek", "merakla sormak", 
    "bilmek istemek", "gizemli", "bilmediğini kabul etmek", "soru işaretleriyle", 
    "merak duygusuyla", "ilgiyle", "meraklı bakışlar", "bir çözüm aramak", "gizemli bir hava", 
    "merak uyandıran durum", "merakla beklemek", "gizemli sorular", "merak dolu bir soru", 
    "merakla gözlemek", "bilgi arayışı içinde olmak", "keşfedilmemiş", "merakla gözler", 
    "sürekli sorular sormak", "hayal kurmak", "gizemli olaylar", "merakla düşünmek", 
    "merakla keşfetmek", "merak içinde düşünmek", "merakla sorgulamak", "meraklı bir kişi", 
    "bilgiye aç", "merakla gözleri açmak", "ilginç bulmak", "merak duygusu uyandıran", 
    "öğrenmeye hevesli", "merakla yol almak", "merak içinde kalmak", "merakla bakmak", 
    "merak dolu bir an", "merak içinde olmak"
]
}
# Kullanıcılar için bir sözlük
users = {}
# Mikrofon kaydını dinlemek için
last_prediction_time = 0  # Son tahminin yapıldığı zaman
prediction_interval = 1.5  # Her 1.5 saniyede bir tahmin yap

# Başlangıç sayfası
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    
def find_root(word):
    results = morphology.analyze(word)
    for result in results:
        # Kök kelimeyi al
        if result.get_stem():  
            return result.get_stem()
    return word   

def analyze_emotions(sentence):
    # Cümledeki kelimeleri temizle ve ayır
    words = re.findall(r'\b\w+\b', sentence.lower())
    
    # Kökleri tespit et
    roots = [find_root(word) for word in words]
    
    # Her duygu kategorisi için kelime sayılarını hesapla
    emotion_counts = defaultdict(int)
    for root in roots:
        for emotion, keywords in emotions.items():
            if root in keywords:
                emotion_counts[emotion] += 1
    if not emotion_counts:
        return {"Nötr": 100}

    # Toplam kelime sayısı
    total_keywords = sum(emotion_counts.values())

    # Yüzdelik oranları hesapla ve normalize et
    percentages = {
        emotion: (count / total_keywords * 100) if total_keywords > 0 else 0
        for emotion, count in emotion_counts.items()
    }

    return percentages

# Kategori tahmini
def predict_category(text):
    text = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(word in text for word in keywords):
            return category
    return "Kategori Bulunamadı"    
    
# Mikrofon kaydını başlatan fonksiyon
def audio_callback(indata, frames, time_info, status):
    global last_prediction_time
    
    if status:
        print(status)
    
    audio_data = indata[:, 0]  # Sadece tek kanal alıyoruz (mono)
    
    # Gürültü azaltma işlemi
    audio_data = nr.reduce_noise(y=audio_data, sr=16000)  # Gürültüyü azalt
    
    features = extract_features(audio_data)  # Use the correct feature extraction function
    
    if features is not None:
        # Zaman bilgisini 'time.time()' ile alıyoruz
        current_time = time.time()  # Global zaman bilgisi
        
        if current_time - last_prediction_time >= prediction_interval:
            # Belirli bir zaman aralığından sonra tahmin yap
            prediction = model.predict([features])[0]
            socketio.emit('speaker_update', {'speaker': prediction})
            last_prediction_time = current_time
            
def record_audio(file_path):
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)  # Sample rate

        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
            print("Recording...")
            time.sleep(10)  # 10 seconds of recording
            print("Recording finished.")            
   
# Mikrofon kaydını başlatma
recording_thread = None
is_recording = False

@socketio.on('start_recording')
def start_recording():
    global recording_thread, is_recording
    if not is_recording or (recording_thread and not recording_thread.is_alive()):
        # Ses verisini sürekli olarak dinlemek için mikrofonu başlatıyoruz
        print("Recording started...")
        is_recording = True
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
    else:
        print("Recording already in progress or thread is active.")

@socketio.on('stop_recording')
def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped")
    socketio.emit('speaker_update', {'speaker': 'None'})
    
# Yeni: Analyze route (manuel metin gönderimi için)
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # 1. Gelen JSON verisini al
        data = request.json
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "Boş metin gönderildi"}), 400
        print(f"Gelen metin: {text}")

        # 2. Kategori tahmini
        category = predict_category(text)
        print(f"Tahmin edilen kategori: {category}")
        #duygu tahmini
        emotion = analyze_emotions(text)
        print("\nDuygu Analizi Sonuçları:")
        for emotion, percentage in emotion.items():
            print(f"{emotion}: %{percentage:.2f}")
            
        # Yanıtı döndür
        return jsonify({
            "speaker": "Bilinmiyor",  # Bu route'da konuşmacı tahmini yapılmıyor
            "category": category,
            "emotion": emotion
        })

    except Exception as e:
        print(f"Sunucu hatası: {e}")
        return jsonify({"error": "Sunucu hatası", "details": str(e)}), 500    
    
@socketio.on('make_prediction')
def make_prediction(data):
    user_name = data.get('userName')
    print(f"Prediction for user {user_name} started.")

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
            while is_recording:
                time.sleep(1)  # Wait for 1 second before processing new audio
    except Exception as e:
        print(f"Error during real-time prediction: {e}")    

def record_audio():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        while is_recording:
            time.sleep(1)  # 1 saniye bekle, sonra ses verisini al ve tahmin yap
    
    
# Kullanıcı ekleme
@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form.get('username')
    if username:
        file_path = os.path.join(UPLOAD_FOLDER, f"{username}.wav")
        if os.path.exists(file_path):
            return jsonify({"status": "error", "message": "Kullanıcı adı zaten mevcut."})
        else:
            users[username] = []  # Kullanıcıyı ekle
            return jsonify({"status": "success", "message": f"Kullanıcı {username} kaydedildi."})
    return jsonify({"status": "error", "message": "Kullanıcı adı gerekli."})

# Ses kaydını kaydetme ve işleme
@app.route('/save_audio', methods=['POST'])
def save_audio():
    username = request.form.get('username')
    audio_file = request.files.get('audio')

    if username and audio_file:
        file_path = os.path.join(UPLOAD_FOLDER, f"{username}.webm")

        if os.path.exists(file_path):
            return jsonify({"status": "error", "message": "Bu kullanıcı adı için ses kaydı mevcut."})

        try:
            with open(file_path, 'wb') as f:
                f.write(audio_file.read())

            # WebM dosyasını WAV'e dönüştür
            audio = AudioSegment.from_file(file_path, format="webm")
            wav_path = os.path.join(UPLOAD_FOLDER, f"{username}.wav")
            audio.export(wav_path, format="wav")
            os.remove(file_path)

            user_folder = os.path.join(UPLOAD_FOLDER, username)
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            # 1.5 saniyelik parçalara ayır
            split_and_augment_audio(wav_path, user_folder)

            # Waveform ve spectrogram görsellerini oluştur
            waveform_data = generate_waveform(wav_path)
            spectrogram_data = generate_spectrogram(wav_path)

            return jsonify({
                "status": "success",
                "message": f"Audio {username} için kaydedildi ve WAV formatına dönüştürüldü.",
                "waveform": waveform_data,
                "spectrogram": spectrogram_data
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

@app.route('/process', methods=['POST'])
def process():
    csv_file = '../features.csv'
    
    # CSV dosyasını kontrol et ve boşalt
    if os.path.exists(csv_file):
        with open(csv_file, 'w') as file:
            file.truncate(0)  # Dosyayı boşalt
    
    # Verileri işleyip CSV'ye kaydet
    process_audio_files(csv_file)
    train_model()
    
    return "Ses dosyaları işlendi ve CSV'ye kaydedildi!"

# Tahmin yapma endpoint
def train_model():
    try:
        # CSV dosyasındaki verileri oku
        data = pd.read_csv('../features.csv')
        
        # Verileri ayır (özellikler ve etiketler)
        X = data.drop(columns=['label'])
        y = data['label']
        
        # Veriyi eğitim ve test olarak ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modeli eğit
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Modelin doğruluğunu test et
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Modeli kaydet
        joblib.dump(model, '../Model/model.pkl')
        
        print(f'Model başarıyla eğitildi. Doğruluk: {accuracy:.2f}')
        
    except Exception as e:
        print(f"Model eğitme hatası: {str(e)}")

@socketio.on('receive_audio')
def handle_audio(audio_data):
    try:
        # Base64 formatında ses verisi
        audio_data = audio_data.split(',')[1]
        audio_bytes = base64.b64decode(audio_data)

        # Ses verisini wav dosyasına dönüştür
        audio_io = io.BytesIO(audio_bytes)
        wav_data, sr = librosa.load(audio_io, sr=None)

        # Özellikleri çıkar
        features = extract_features(wav_data, sr)

        # Tahmin yap
        prediction = model.predict([features])[0]

        # Tahmin sonucunu geri gönder
        emit('prediction_result', {'prediction': prediction})
    except Exception as e:
        print(f'Hata: {str(e)}')
        emit('prediction_result', {'prediction': 'Hata oluştu'})


# Ses dosyasını 1.5 saniyelik parçalara ayırma
def split_and_augment_audio(input_path, output_dir, segment_duration_ms=1500):
    try:
        audio = AudioSegment.from_file(input_path)
        for i, start_time in enumerate(range(0, len(audio), segment_duration_ms)):
            segment = audio[start_time:start_time + segment_duration_ms]
            augmented_segments = augment_audio_segment(segment)
            for j, augmented_segment in enumerate(augmented_segments):
                segment_filename = os.path.join(output_dir, f"augmented_segment_{i + 1}_{j + 1}.wav")
                augmented_segment.export(segment_filename, format="wav")
    except Exception as e:
        print(f"Hata: {str(e)}")

# Veri arttırma
def augment_audio_segment(segment):
    augmented_audio = []
    augmented_audio.append(change_speed(segment))
    augmented_audio.append(change_pitch(segment))
    augmented_audio.append(add_noise(segment))
    augmented_audio.append(duplicate_audio(segment))
    return augmented_audio

# Sesin hızını değiştirme
def change_speed(segment, speed_factor=None):
    if speed_factor is None:
        speed_factor = random.uniform(0.7, 1.5)
    return segment.speedup(playback_speed=speed_factor)

# Sesin tonunu değiştirme
def change_pitch(segment, pitch_factor=None):
    if pitch_factor is None:
        pitch_factor = random.uniform(-2, 2)
    return segment._spawn(segment.raw_data, overrides={
        "frame_rate": int(segment.frame_rate * (2 ** pitch_factor))
    }).set_frame_rate(segment.frame_rate)

# Gürültü ekleme
def add_noise(segment, noise_factor=None):
    if noise_factor is None:
        noise_factor = random.uniform(0.01, 0.05)
    samples = np.array(segment.get_array_of_samples())
    noise = np.random.normal(0, noise_factor, samples.shape)
    noisy_samples = samples + noise
    noisy_samples = np.clip(noisy_samples, -2**15, 2**15-1)
    noisy_segment = segment._spawn(noisy_samples.astype(np.int16).tobytes())
    return noisy_segment

# Sesin çoğaltılması
def duplicate_audio(segment, num_duplicates=2):
    return segment * num_duplicates

# Özellikleri çıkarma
def extract_features_csv(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Ses dosyasını yükle
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # İlk 13 MFCC
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Chroma
        rms = librosa.feature.rms(y=y)[0]  # RMS
        zcr = librosa.feature.zero_crossing_rate(y)[0]  # Zero Crossing Rate

        # Özellikleri düzleştir
        features = list(mfccs.mean(axis=1)) + list(chroma.mean(axis=1)) + [rms.mean(), zcr.mean()]
        return features
    except Exception as e:
        print(f"Özellik çıkarma hatası: {file_path} -> {str(e)}")
        return None
    
def extract_features(audio_data, sr=16000):
    try:
        # Use librosa to handle raw audio data
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        rms = librosa.feature.rms(y=audio_data)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        features = list(mfccs.mean(axis=1)) + list(chroma.mean(axis=1)) + [rms.mean(), zcr.mean()]
        return features
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None
   

# CSV'ye yazma fonksiyonu
# Özellikleri ve etiketi ekle
def update_csv(csv_file, features, label):
    try:
        # Dosyanın mevcut olup olmadığını kontrol et
        file_exists = os.path.exists(csv_file)

        # Dosyayı aç
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Dosya yoksa veya içerik boşsa başlıkları yaz
            if not file_exists or os.stat(csv_file).st_size == 0:
                header = [
                    'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9',
                    'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13', 'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4',
                    'Chroma_5', 'Chroma_6', 'Chroma_7', 'Chroma_8', 'Chroma_9', 'Chroma_10', 'Chroma_11',
                    'Chroma_12', 'RMS', 'Zero_Crossing_Rate', 'label'
                ]
                writer.writerow(header)  # Başlıkları dosyaya yaz

            # Özellikleri ve etiketi ekle
            writer.writerow(features + [label])  # Verileri dosyaya yaz
    except Exception as e:
        print(f"CSV yazma hatası: {csv_file} -> {str(e)}")

# Ses dosyalarını işle
def process_audio_files(csv_file):
    audio_folder = '../Audio'  # Ana Audio klasörü
    try:
        for label in os.listdir(audio_folder):
            label_folder = os.path.join(audio_folder, label)
            if os.path.isdir(label_folder):  # Alt klasör mü?
                for file in os.listdir(label_folder):
                    if file.endswith(".wav"):  # Sadece .wav dosyalarını işle
                        file_path = os.path.join(label_folder, file)
                        print(f"İşleniyor: {file_path}")
                        features = extract_features_csv(file_path)
                        if features:
                            update_csv(csv_file, features, label)
    except Exception as e:
        print(f"Ses dosyaları işlenirken hata oluştu: {str(e)}")




def delete_existing_images(image_folder):
    try:
        # Görsel dosyalarını kontrol et
        for file in os.listdir(image_folder):
            file_path = os.path.join(image_folder, file)
            if file.endswith('.png') and os.path.isfile(file_path):
                os.remove(file_path)  # Görsel dosyasını sil
                print(f"Görsel dosyası silindi: {file}")
    except Exception as e:
        print(f"Görsel dosyaları silinirken hata oluştu: {str(e)}")
        


# Görsel oluşturma
def generate_waveform(wav_path):
    samplerate, data = wavfile.read(wav_path)
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(data) / samplerate, num=len(data)), data)
    plt.title('Waveform')
    plt.xlabel('Zaman [s]')
    plt.ylabel('Genlik')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    waveform_data = base64.b64encode(img.read()).decode('utf-8')
    plt.close()  # Kaynakları temizle
    return waveform_data

def generate_spectrogram(wav_path):
    samplerate, data = wavfile.read(wav_path)
    f, t, Sxx = spectrogram(data, samplerate)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, np.log(Sxx), shading='auto')
    plt.title('Spectrogram')
    plt.ylabel('Frekans [Hz]')
    plt.xlabel('Zaman [s]')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    spectrogram_data = base64.b64encode(img.read()).decode('utf-8')
    plt.close()  # Kaynakları temizle
    return spectrogram_data

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)