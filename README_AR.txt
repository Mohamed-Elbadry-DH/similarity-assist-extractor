الملف ده جاهز عشان تعمل Deploy على Render (Extractor Service).

الخدمة دي بتوفر 3 Endpoints أساسية:
- GET  /health   (يتأكد إن الخدمة شغالة وOCR متاحة)
- POST /features (استخراج Features + OCR من صورة لوجو)
- POST /rerank   (Visual Reranking: SSIM + ORB لأفضل المرشحين)

========================
A) لو أول مرة تعمل Deploy
========================

1) افتح GitHub واعمل Repository جديد اسمه مثلا:
   similarity-assist-extractor

2) ارفع فيه الملفات دي:
   - main.py
   - requirements.txt
   - Dockerfile
   - README_AR.txt

3) افتح Render
4) اختار New → Web Service
5) اختار GitHub Repository
6) اختار الـrepo اللي رفعته
7) Render هيكتشف Dockerfile تلقائيا
8) اضغط Create Web Service / Deploy

بعد ما يخلص، هيديك رابط مثل:
  https://something.onrender.com

اختبر في المتصفح:
  https://something.onrender.com/health
لازم يرجع JSON فيه:
  status = ok

========================
B) لو أنت عامل Deploy بالفعل وعايز تحدثه
========================

1) افتح نفس GitHub repo بتاع extractor
2) استبدل الملفات (main.py و requirements.txt و README_AR.txt) بالنسخة الجديدة
3) اعمل Commit
4) Render هيعمل Deploy تلقائي (Auto Deploy)

بعد ما يخلص Deploy، اختبر:
  https://something.onrender.com/health

========================
C) تستخدمه داخل v0
========================

في v0 حط Environment Variable اسمها:
  EXTRACTOR_BASE_URL
وقيمتها تكون الرابط الأساسي فقط، مثال:
  https://something.onrender.com

مهم جدا:
- لا تضع /health
- لا تضع /features
- لا تضع /rerank
- فقط الرابط الأساسي

========================
D) ملاحظات سريعة
========================

- Endpoint /rerank يتوقع JSON فيه query_image (base64) و candidates (logo_url لكل سجل).
- v0 هو اللي هيستدعي /features أولاً ثم /rerank لأفضل 20-30 مرشح.
