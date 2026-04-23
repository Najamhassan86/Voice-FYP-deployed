import React from 'react';
import { Link } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';
import { ArrowRight, Zap, Users, Smile, BookOpen, Sparkles, ChevronRight, Rocket } from 'lucide-react';

export const Home = () => {
  const { t, language } = useLanguage();

  const stats = [
    { value: '3', labelEn: 'Core Modes', labelUr: 'بنیادی موڈز' },
    { value: '32+', labelEn: 'PSL Animations', labelUr: 'PSL اینیمیشنز' },
    { value: 'Real-time', labelEn: 'Inference', labelUr: 'فوری شناخت' },
  ];

  const features = [
    {
      title: t('home_feature1'),
      descEn: 'Live hand landmark tracking and sequence recognition for practical PSL translation.',
      descUr: 'PSL ترجمہ کے لیے براہ راست ہینڈ لینڈمارک ٹریکنگ اور سیکوئنس شناخت۔',
      icon: Zap,
    },
    {
      title: t('home_feature2'),
      descEn: 'Urdu and English output with clear reading flow and voice playback support.',
      descUr: 'واضح ریڈنگ فلو اور آواز کے ساتھ اردو اور انگریزی آؤٹ پٹ۔',
      icon: Users,
    },
    {
      title: t('home_feature3'),
      descEn: 'Text-to-sign rendering through a 3D animated avatar pipeline.',
      descUr: '3D اینیمیٹڈ اوتار کے ذریعے ٹیکسٹ کو سائن میں تبدیل کریں۔',
      icon: Smile,
    },
    {
      title: t('home_feature4'),
      descEn: 'Practice mode with scoring feedback to improve your signing consistency.',
      descUr: 'اسکورنگ فیڈ بیک کے ساتھ پریکٹس موڈ جو کارکردگی بہتر کرے۔',
      icon: BookOpen,
    },
  ];

  return (
    <div className="pt-8">
      <section className="page-wrap reveal-up reveal-delay-1">
        <div className="section-card relative overflow-hidden p-8 sm:p-12">
          <div className="absolute -right-20 -top-20 h-72 w-72 rounded-full bg-[#4f46e5]/25 blur-3xl" />
          <div className="absolute -bottom-24 -left-20 h-72 w-72 rounded-full bg-[#ff8a3d]/25 blur-3xl" />
          <div className="absolute left-[38%] top-10 h-48 w-48 rounded-full bg-[#2ea6ff]/18 blur-3xl" />

          <div className="relative grid items-center gap-10 lg:grid-cols-[1.2fr_0.8fr]">
            <div>
              <p className="eyebrow mb-4">
                <Sparkles className="h-4 w-4" />
                {language === 'en' ? 'Human-Centered AI Interface' : 'انسانی مرکز AI انٹرفیس'}
              </p>

              <h1 className={`title-display text-5xl sm:text-6xl ${language === 'ur' ? 'ur-text' : ''}`}>
                {t('home_subtitle')}
              </h1>

              <p className={`mt-5 max-w-2xl text-lg text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
                {t('home_tagline')}
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <Link to="/psl-to-text" className="btn-primary">
                  {t('home_start')} <ArrowRight className="h-4 w-4" />
                </Link>
                <Link to="/text-to-psl" className="btn-secondary">
                  {t('nav_text_to_psl')} <ChevronRight className="h-4 w-4" />
                </Link>
              </div>

              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                {stats.map((item) => (
                  <div key={item.value} className="vivid-panel p-4">
                    <p className="text-2xl font-semibold text-[#1b2a45]">{item.value}</p>
                    <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en' ? item.labelEn : item.labelUr}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-[#cad8f1] bg-gradient-to-br from-white to-[#eef5ff] p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-[#5f73a5]">VOICE</p>
                  <p className="text-lg font-semibold text-[#1a2642]">Communication Studio</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-[#ff8a3d] to-[#ff5d5d] text-white">
                  <Rocket className="h-5 w-5" />
                </div>
              </div>

              <div className="space-y-4">
                <div className="rounded-2xl border border-[#d9e3f5] bg-white p-4">
                  <p className="text-sm font-medium text-[#2a4035]">PSL Input Stream</p>
                  <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[#e9f0ff]">
                    <div className="h-full w-3/4 rounded-full bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86]" />
                  </div>
                </div>

                <div className="rounded-2xl border border-[#d9e3f5] bg-white p-4">
                  <p className="text-sm font-medium text-[#2a4035]">Text + Voice Output</p>
                  <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[#ffe9da]">
                    <div className="h-full w-2/3 rounded-full bg-gradient-to-r from-[#ff8a3d] to-[#ff5d5d]" />
                  </div>
                </div>

                <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
                  {language === 'en'
                    ? 'A modern, production-ready interface crafted for accessibility and clarity.'
                    : 'رسائی اور وضاحت کے لیے تیار کردہ جدید اور پیشہ ور انٹرفیس۔'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="page-wrap mt-10 reveal-up reveal-delay-2">
        <div className="section-card p-8 sm:p-10">
          <div className="mb-8 flex items-end justify-between gap-4">
            <div>
              <p className="eyebrow mb-2">Platform Highlights</p>
              <h2 className={`title-display text-3xl ${language === 'ur' ? 'ur-text' : ''}`}>
                {t('home_features')}
              </h2>
            </div>
            <span className="hidden rounded-full border border-[#cad8f1] bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#5f73a5] sm:inline-flex">
              VOICE Suite
            </span>
          </div>

          <div className="stagger-grid grid gap-4 md:grid-cols-2">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <div key={feature.title} className="rounded-2xl border border-[#d5e0f2] bg-white/92 p-6">
                  <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-[#ecebff] to-[#e1f2ff] text-[#4f46e5]">
                    <Icon className="h-5 w-5" />
                  </div>
                  <h3 className={`mb-2 text-lg font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {feature.title}
                  </h3>
                  <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? feature.descEn : feature.descUr}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="page-wrap mt-10 reveal-up reveal-delay-2">
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="section-card p-8">
            <p className="eyebrow mb-2">Workflow Modes</p>
            <h2 className={`title-display mb-3 text-3xl ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en' ? 'Choose Your Translation Path' : 'اپنا ترجمہ کا راستہ منتخب کریں'}
            </h2>
            <p className={`mb-6 text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en'
                ? 'Switch between live PSL recognition, text-to-sign avatar rendering, and guided sign practice.'
                : 'لائیو PSL شناخت، ٹیکسٹ سے سائن اوتار، اور گائیڈڈ پریکٹس کے درمیان آسانی سے سوئچ کریں۔'}
            </p>
            <div className="flex flex-wrap gap-3">
              <Link to="/psl-to-text" className="btn-primary">{t('nav_psl_to_text')}</Link>
              <Link to="/text-to-psl" className="btn-secondary">{t('nav_text_to_psl')}</Link>
              <Link to="/learn" className="btn-secondary">{t('nav_learn')}</Link>
            </div>
          </div>

          <div className="section-card p-8">
            <p className="eyebrow mb-2">Team</p>
            <h2 className={`title-display mb-5 text-3xl ${language === 'ur' ? 'ur-text' : ''}`}>
              {t('home_team')}
            </h2>
            <div className="stagger-grid grid gap-3 sm:grid-cols-2">
              {[['Shaheer Zaman', '22I-0805'], ['Najam Hassan', '22I-1332'], ['Aliyan Zafar', '20I-2414']].map(([name, id]) => (
                <div key={name} className="rounded-2xl border border-[#d5e0f2] bg-white px-4 py-3">
                  <p className="font-semibold text-[#1a2642]">{name}</p>
                  <p className="text-sm text-[#5c6d64]">{id}</p>
                </div>
              ))}
            </div>
            <p className={`mt-4 text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en' ? 'Supervisor: Mr. Almas Khan' : 'نگرانی کے تحت: Mr. Almas Khan'}
            </p>
          </div>
        </div>
      </section>

      <section className="page-wrap mt-10 reveal-up reveal-delay-3">
        <div className="section-card p-8">
          <h2 className={`title-display mb-2 text-2xl ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en' ? 'Project Resources' : 'پروجیکٹ وسائل'}
          </h2>
          <p className={`mb-5 text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en' ? 'Explore source code, setup guides, and documentation.' : 'سورس کوڈ، سیٹ اپ گائیڈز اور دستاویزات ملاحظہ کریں۔'}
          </p>
          <div className="stagger-grid grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ['GitHub', 'https://github.com/Najamhassan86/VOICE-FYP'],
              ['README', 'https://github.com/Najamhassan86/VOICE-FYP/blob/main/README.md'],
              ['License', 'https://github.com/Najamhassan86/VOICE-FYP/blob/main/LICENSE'],
              ['Contact', 'tel:+923171975525'],
            ].map(([label, href]) => (
              <a
                key={label}
                href={href}
                target={href.startsWith('http') ? '_blank' : undefined}
                rel={href.startsWith('http') ? 'noopener noreferrer' : undefined}
                className="rounded-2xl border border-[#d5e0f2] bg-white px-4 py-4 text-sm font-medium text-[#2d3c54] transition-colors hover:bg-[#edf3ff]"
              >
                {label}
              </a>
            ))}
          </div>
        </div>
      </section>

      <section className="page-wrap mt-10">
        <div className="rounded-3xl border border-[#cad8f1] bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] p-8 text-white sm:p-10">
          <h2 className={`title-display text-3xl text-white ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en' ? 'Start Building Better Conversations Today' : 'آج ہی بہتر گفتگو کی شروعات کریں'}
          </h2>
          <p className={`mt-3 max-w-2xl text-sm text-emerald-50 ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en'
              ? 'Launch live sign recognition, synthesize speech, and make every interaction more accessible.'
              : 'لائیو سائن شناخت شروع کریں، آواز تخلیق کریں، اور ہر رابطے کو مزید قابلِ رسائی بنائیں۔'}
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link to="/psl-to-text" className="rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-[#0f5f3f]">
              {t('nav_psl_to_text')}
            </Link>
            <Link to="/text-to-psl" className="rounded-full border border-white/50 px-5 py-2.5 text-sm font-semibold text-white">
              {t('nav_text_to_psl')}
            </Link>
          </div>
        </div>
      </section>

      <section className="h-10" />
    </div>
  );
};
