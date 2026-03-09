الملف ده جاهز عشان تعمله Deploy على Render.

اعمل الآتي فقط:

1) افتح GitHub واعمل Repository جديد اسمه مثلا:
   similarity-assist-extractor

2) ارفع فيه الملفات الأربعة الموجودة هنا:
   - main.py
   - requirements.txt
   - Dockerfile
   - README_AR.txt

3) افتح Render
4) اختار New + ثم Web Service
5) اختار GitHub Repository
6) اختار الـrepo اللي رفعته
7) Render هيكتشف Dockerfile تلقائيا
8) اضغط Create Web Service أو Deploy

بعد ما يخلص:
- هيديك رابط مثل:
  https://something.onrender.com

اختبر الرابط ده في المتصفح:
- افتح:
  https://something.onrender.com/health

لو ظهر JSON فيه status = ok
يبقى الخدمة شغالة.

بعدها:
- ارجع إلى v0
- في EXTRACTOR_BASE_URL حط الرابط الأساسي فقط
مثال:
  https://something.onrender.com

مهم:
- لا تضع /health
- لا تضع /features
- فقط الرابط الأساسي
