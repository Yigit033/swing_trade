# Trading Profitability Audit

**11 Ekim 2026 — Sonuç: mevcut ürünün pozitif NET işlem avantajı kanıtlanmış değil.**

Production değiştirilmedi. Güncel PostgreSQL kayıtları `readonly=True` bağlantıyla, yalnız SELECT kullanılarak okundu; işlemler rollback/close ile sonlandırıldı. Fiyat kontrolünde mevcut immutable Phase 2 cache’i kullanıldı; yeni piyasa verisi indirilmedi. İncelenen yerel HEAD: `9b3bebf3a12813ce172882f9d43cf997fd70363c`; mevcut çalışma ağacı değişiklikleri korundu. Uzak deployment’ın aynı kod olduğu doğrulanmış değildir.

## 1. Gerçekten pozitif net avantaj var mı?

**Henüz gösterilemiyor.** 25 Mart–9 Ekim 2026 arasında 40 kapanmış paper işlem, 3 reddedilmiş kayıt, 0 açık/bekleyen işlem var. Broker execution kaydı yok. İki ayrı kullanıcı/hesap kapsamı ve farklı tarihsel kurallar mevcut; aşağıdaki kapsamlar da tek strateji sürümü sayılmaz.

| Ayrı paper kayıt kapsamı | İşlem | Giriş–son çıkış dönemi | Maliyet öncesi PnL | Dolar PF | 50 bps maliyetle PnL |
|---|---:|---|---:|---:|---:|
| A | 29 | 25 Mart–2 Eylül | −$424,18 | 0,813 | −$901,06 |
| B | 11 | 3 Ağustos–9 Ekim | +$496,76 | 2,303 | +$424,79 |

**Eski PF≈1,03’ün muhasebe mutabakatı:** iki kapsamın ve eski/yeni kuralların birlikte toplanması +$72,58, işlem başına +$1,81, %50 kazanma oranı ve **PF=1,02736** veriyor. Bu birleşim yalnız önceki rakamı doğrular; mevcut stratejinin performansı veya tek portföy getirisi değildir. Yüzde getiriler üzerinden PF ise **1,529**: değişken pozisyon boyutları nedeniyle dolar PF ile aynı metrik değildir.

| Birleşik eski rakamın maliyet duyarlılığı; strateji testi değildir | Toplam PnL | İşlem başına PnL | Dolar PF |
|---|---:|---:|---:|
| Kayıtlı; spread/komisyon kesintisi yok | +$72,58 | +$1,81 | 1,027 |
| 20 bps toplam maliyet | −$146,96 | −$3,67 | 0,947 |
| 50 bps toplam maliyet | −$476,27 | −$11,91 | 0,838 |
| 100 bps toplam maliyet | −$1.025,12 | −$25,63 | 0,692 |

Maliyet senaryosu: her işlemde `kayıtlı PnL − giriş fiyatı × başlangıç adedi × bps/10.000`; **gidiş-dönüş toplam** varsayım, ölçülmüş spread değildir. Kayıtlı gap zararı zaten PnL’dedir. Eşit ağırlıklı ortalama fiyat getirisi +%1,5275, medyan −%0,46; bunlar portföy getirisi değildir. En büyük kazanan LIFE (+$431,32) çıkarılınca birleşik brüt PnL −$358,74, PF 0,865 oluyor. B kapsamındaki brüt kârın %86,8’i LIFE’tan geliyor.

**Negatif average R doğrulandı ama tam örneklemi temsil etmiyor:** başlangıç stop mesafesi pozitif kalan **31/40** kayıtta ortalama **−0,329R**, medyan **−0,756R**. T1 uygulanmış diğer **9 işlemin tamamı kazanan**; `initial_stop=entry_price` yapılmış, özgün risk kaybolmuş. Bu kazananları dışlayan R ortalamasına “sistemin expectancy’si” denemez. 40 işlemin güvenilir average/median R’si **hesaplanamıyor**. Kaynak: [tracker.py — update_trade_status](swing_trader/paper_trading/tracker.py#L503), [storage.py — close_trade](swing_trader/paper_trading/storage.py#L615). Kayıtlı PnL’nin kısmi adet yuvarlaması dahil mevcut muhasebe formülüyle mutabakatı: **40/40**; bu, fill’lerin gerçekliğini doğrulamaz.

## 2. Hangi sonuçlar güvenilir, hangileri yetersiz?

### Production davranışı ve yeniden oynatma

- **Seçim:** aktif VCE/RVOL → hard gate/confirmation/stage → Quality eşiği; Rank yalnız sıralama. [signals.py — check_all_triggers](swing_trader/small_cap/signals.py#L825), [engine.py — scan_stock / scan_universe](swing_trader/small_cap/engine.py#L578). Eski düşük skorlu işlemler bugünkü politika altında otomatik olarak geçerli sayılmadı.
- **Giriş/boyut:** paper confirmation sonraki mevcut seans Open’ını kullanıyor; efektif gap limitleri +%5/−%7. Stop 2,5 ATR, C/A/B genişlik tavanları %14/%15/%16. Takibe eklerken risk bütçesi %1,5, pozisyon tavanı C/A %25, B %20; confirmation fiyat/stop’u değiştiriyor fakat adedi yeniden hesaplamıyor. [tracker.py — confirm_pending_trades](swing_trader/paper_trading/tracker.py#L732), [risk.py — calculate_position_size](swing_trader/small_cap/risk.py#L189), [scanner.py — _authoritative_position_size](api/routers/scanner.py#L92). Ayar patch tarihi: **9 Eylül 2026 16:43:19Z**.
- **Çıkış:** stop önce; gap stop altına açılırsa Open. T1’de efektif %33 kısmi satış, stop başa başa, hedef T2’ye geçiyor. Hedefler tip/ATR/Quality/rejimden dinamik; chandelier geçmiş tepe−3 ATR, aktivasyon +1,5 ATR. Timeout işlem kaydındaki `max_hold_days` ile zorunlu çıkış; bütün işlemlere tek bir “20 gün” uygulanmıyor. [risk.py — calculate_targets](swing_trader/small_cap/risk.py#L124), [tracker.py — check_exit_conditions](swing_trader/paper_trading/tracker.py#L280).
- **Kanıtlanmış zamanlama hataları:** fonksiyon `price_history.iloc[1:]` ile giriş gününü atlıyor. Sentetik kontrolde giriş gününde 90 stop’unun altında 85 Low görülmesine rağmen işlem OPEN kaldı. T1 sonrası `initial_stop` başa başa yazılıp tüm geçmiş yeniden oynatılıyor: ikinci kontrolde **5 Ekim T1 → 2 Ekim stop** üretildi. Bunlar mevcut fonksiyonla, depolama/API çağrısı olmadan yeniden üretildi; tarihsel dolar etkileri henüz ayrıştırılamıyor.
- **Gerçek kayıt kontrolü:** cache **38/40** işlemi kapsıyor; UMAC/CPRX yok. 9 kısmi satışın **5’inde** kayıtlı T1, giriş–kayıtlı çıkış arasındaki en yüksek fiyatın üzerinde: PRCT 28,37>28,14; PAR 15,91>15,40; BRUN 27,43>26,75; PUBM 10,83>10,24; MITK 17,44>17,27. Nominal/split kontrolü ve adjusted OHLC karşılaştırması uyuşmazlığı kaldırmadı. **Fill kaydı ile güncel fiyat geçmişi tutarsız; asıl neden veri revizyonu mu, eski motor mu, manuel düzenleme mi kesin değil.**
- Ağustos–Ekim’de kısmi satışı ve kayıtlı stop düzenlemesi olmayan **10 işlem**, mevcut çıkış fonksiyonuyla kayıtlı çıkış tarihine kadar oynatıldı: **9/10 durum+tarih**, **8/10 fiyat ±$0,01** eşleşti. KYMR, 17 Ağustos kaydında TRAILED/+ $156,66 iken replay OPEN; SAIC’te dividend-adjusted fiyat farkı var. Bu nedenle aynı sürüm, aynı fiyat tabanı ve değişmez emir geçmişi olmadan “production-birebir yeni PnL” üretmedim.

### Backtest ve canlı sinyal sonuçları ayrı

**Arşiv backtest:** [backtest_live_replica.json](output/backtest_live_replica.json) eski/yeni exit için ayrı ayrı **469** sinyal bildiriyor: ortalama +%0,724 / +%1,290; medyan −%0,922 / +%0,129; kazanma %44,99 / %50,53. Bunlar yeniden çalıştırılmış güncel sonuç değil, mevcut arşivdir. Cache 995 ticker, **3 Haziran 2024–29 Mayıs 2026**; bugünkü döngünün potansiyel sinyal aralığı **28 Ağustos 2024–29 Nisan 2026**. Sinyal bazlı kayıt/sürüm manifesti olmadığı için arşivin tam fiili ilk/son işlem tarihi doğrulanamıyor.

“Canlı-birebir” iddiası **yanlış**: [simulate / EXIT_NEW](scripts/backtest_live_replica.py#L124) sabit %10 T1, T2 yok, 20 günlük hold kullanıyor; production dinamik çift hedef kullanıyor. Döngü entry dahil 21 bara kadar gidebiliyor, paper giriş barını atlıyor. Güncel shares/float geçmişe taşınıyor, ticker evreni point-in-time değil; örtüşen sinyaller ayrı sayılıyor. `_slippage_bps` sinyal tarihine kırpmadan `df.tail(20)` kullanıyor: **gelecek likiditesi maliyet hesabına sızıyor**. Arşivde “OOS” yazması bağımsız doğrulama sağlamaz. Diğer 5.660/8.719 işlemli eski/yeni özetler farklı seçim kurallarıdır; birleştirilmedi.

Ürünün ayrı [SmallCapBacktest._process_pending_entries](swing_trader/small_cap/smallcap_backtest.py#L443) yolu da Open’dan giriş kararı için **giriş gününün tamamlanmış OHLC’siyle ATR/stop/target/sizing** hesaplıyor. Bu açık look-ahead’dır; paper ile eşdeğer değildir. Bu yoldan sürümü sabitlenmiş, doğrulanabilir işlem defteri bulunmadığından yeni PF/DD iddiası yok.

**Canlı oluşmuş sinyal ≠ işlem:** 10 Haziran–18 Eylül’de 34 olgun kayıt var; 33’ü `forward.run_id = runs.job_id` ile eşleşiyor, **5’i kendi taramasının kayıtlı Quality tabanının altında**. Örnek farklı kohortlar: VCE/Q65, 17 Haziran–20 Temmuz **n=12, ortalama r10 −%5,317**; RVOL/Q78, 7–31 Ağustos **n=4, +%5,302**; RVOL/Q80, 3 Ağustos–18 Eylül **n=5, −%1,136**. Aynı eşik dahi aynı kod sürümünü kanıtlamaz; birleştirilmedi. VCE/Q78 grubundaki 4 kaydın 2’si ardışık ASST sinyali; bağımsız fırsat sayılmaz.

Yerel cache ile **33/34** forward sonuç yeniden hesaplandı: **31’inde fark ≤0,10 yüzde puanı**; PRCH +%13,78→+%17,99, SEI +%7,12→+%3,04. CPRX değerlendirilemedi. Kaynak [forward_returns.py — _compute_forward_values](swing_trader/data/forward_returns.py#L345): next-open→10. seans Close; **T1/T2, stop ve maliyet uygulamaz**, MFE kâr değildir. Bu tablo gerçek işlem avantajı için yeterli değil.

**Benchmark/portföy sınırı:** kısmi satışı olmayan kayıtlarda aynı giriş günü Open/çıkış günü Close ve başlangıç nominaliyle SPY, her iki tarafa 50 bps uygulanarak karşılaştırıldı: A **n=21**, paper −$2.145,85 / SPY −$295,12; B **n=10**, paper −$1,85 / SPY −$68,93. Bu yalnız tarihleri paper işleminden alınan referans; SPY çıkışı Close olduğu için intraday/gap çıkışlarıyla saat eşleşmesi yok, bağımsız strateji/alpha testi değil. IWM cache’i yok.

A’da en fazla **10**, B’de **5** eşzamanlı kayıt; başlangıç nominali üst sınırı A **$42.383,44**, B **$5.220,14**. Kısmi satış tarihleri, değişmez hesap sermayesi, yatırma/çekmeler ve günlük mark-to-market equity yok: **sermaye kullanım yüzdesi ve gerçek maksimum portföy drawdown hesaplanamıyor; %20–30 toleransına uyum kanıtlanmış değil.** Manuel emir gönderme/fill zamanları da yok; sonraki Open ve stop fiyatında dolum garantisi varsayılamaz ([SEC açıklaması](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-15)).

## 3. Para kaybına en fazla katkı yapan doğrulanmış problem ne?

**Sayısal katkı:** STOPPED grubundaki 15 zarar, toplam **−$2.438,84** ile tüm brüt zararların **%91,9’u**. Beş kârlı STOPPED kaydı dahil grubun neti −$1.872,74. TARGET +$757,44; TIMEOUT +$603,25; TRAILED +$584,63 — yalnız muhasebe ayrıştırması, exit alternatiflerinin nedensel testi değil.

En büyük tek zarar **TH, 22 Haziran→2 Temmuz, 773 adet, −$773**: toplam brüt zararın **%29,1’i**. Ancak run **32**, 17 Haziran 21:51Z, **$100.000** portföy ve aynı 773 adedi kaydediyor. Dolayısıyla “$10.000 hesapta kesin aşırı boyut” denemez. Eski rapordaki NSP işlemi güncel tabloda **yok**; eski kayıp rakamı tekrar kullanılmadı. Sunucu boyut tavanı mevcut kodda zaten var.

**Stop’ları suçlamak da kanıtlanmış değil:** kayıtlı çıkıştan sonraki 10. seans Close, CBRL’de +%12,05, DNOW’da +%25,30 iken TH’de −%14,44. T2 sonrası WYFI −%12,72, RILY −%19,29; OUST +%16,09. Timeout sonrası STAA +%18,16, CLBK −%0,51. Çıkışlar bazen korumuş, bazen devam hareketini kaçırmış görünüyor; tepe satışını varsaymadım. Cache/sürüm uyuşmazlıkları nedeniyle bunlar alternatif exit’in kazanacağını ispatlamaz.

**En ciddi kanıtlanmış ürün problemi ölçüm/exit kronolojisi:** T1 bilgisinin geçmişe taşınması ve özgün riskin kaybı. Bunun kaç dolar zarara yol açtığı mevcut kayıtlarla bilinmiyor. “Zararların asıl nedeni stop genişliği/T1 oranı” diyebilecek güvenilir nedensel kanıt yok.

## 4. İlk hangi TEK değişiklik hipotezini test etmeliyiz?

**Yalnız T1 sonrası stop geçişinin kronolojisi:** diğer tüm kurallar sabitken başa baş stop’unu T1’den önceki barlara uygulamayan replay ile mevcut davranışı eşleştirilmiş test etmeliyiz. Parametre, trigger, hedef veya stop mesafesi değiştirmeden. Önce aynı girdide geleceğe dayalı çıkışın sıfırlanması; ardından maliyetli, tarihsel olarak yeniden kurulabilir işlemler ve ayrı ileri paper kayıtlarında net PnL/R ve drawdown karşılaştırması. Eksik eski T1 tarihlerini uydurmamak şartıyla.

**Sınıflandırma: hata VERIFIED; kârlılık artışı INSUFFICIENT DATA.** Bu turda düzeltme veya alternatif exit uygulanmadı; iyileşme kanıtlanmadı.

## 5. Gerçek sermaye riske atmak için yeterli kanıt var mı?

**Hayır.** Pozitif görünen küçük alt örneklem, tek büyük kazanana bağımlılık, değişen kurallar, tutarsız fill geçmişi ve bilinmeyen portföy drawdown nedeniyle yeterli değil. Mevcut kanıt “kesin kaybettirir” demez; **“maliyet sonrası sürdürülebilir kazandırdığı henüz gösterilemedi”** der.

Doğrulama: `python -m pytest tests swing_trader/tests api/tests -q -p no:cacheprovider --basetemp output/profitability_audit_test_tmp` → **393 geçti**, 3 uyarı, 21,73 saniye. Ayrıca yukarıdaki iki izole davranış kontrolü hataları yeniden üretti. Yeni strateji, production değişikliği veya deployment yapılmadı.
