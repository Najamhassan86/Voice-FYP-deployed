import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { AvatarPlayer } from '../components/AvatarPlayer';
import { Languages, Sparkles, Eraser, ChevronRight } from 'lucide-react';

export const TexttoPSL = () => {
  const { t, language } = useLanguage();
  const [text, setText] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [isAnimating, setIsAnimating] = useState(false);

  // Quick words for English
  const quickWordsEnglish = [
    'excited', 'good', 'smart', 'yes',
    'book', 'ready', 'quick', 'important',
    'fantastic', 'healthy', 'peaceful', 'new'
  ];

  // Quick words for Urdu
  const quickWordsUrdu = [
    'خوش',      // excited/happy
    'اچھا',     // good
    'ذہین',     // smart
    'ہاں',      // yes
    'کتاب',     // book
    'تیار',     // ready
    'تیز',      // quick
    'اہم',      // important
    'بہترین',   // fantastic
    'تندرست',   // healthy
    'پرامن',    // peaceful
    'نیا'       // new
  ];

  const handleTranslate = () => {
    if (text.trim()) {
      setIsAnimating(true);
    }
  };

  const handleAnimationStart = () => {
    // Animation started
  };

  const handleAnimationEnd = () => {
    setIsAnimating(false);
  };

  const handleClear = () => {
    setText('');
    setIsAnimating(false);
  };

  const handleQuickWord = (word) => {
    setIsAnimating(false);
    setText(word);
    setTimeout(() => {
      setIsAnimating(true);
    }, 50);
  };

  return (
    <div className="pt-8">
      <div className="page-wrap space-y-8 reveal-up">
        <div className="section-card p-8 sm:p-10">
          <p className="eyebrow mb-3 text-[#5f73a5]">Text to Sign</p>
          <h1 className={`title-display mb-2 text-4xl ${language === 'ur' ? 'ur-text' : ''}`}>
            {t('text_title')}
          </h1>
          <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
            {t('text_subtitle')}
          </p>
        </div>

        <div className="stagger-grid grid gap-8 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-1">
            <div className="section-card space-y-5 p-6">
              <div>
                <label className={`mb-2 block text-sm font-semibold text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`}>
                  {t('text_language')}
                </label>
                <select
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                  className="field-input"
                >
                  <option value="en">English</option>
                  <option value="ur">اردو</option>
                </select>
              </div>

              <div>
                <label className={`mb-2 block text-sm font-semibold text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`}>
                  {language === 'en' ? 'Enter Text' : 'متن درج کریں'}
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={selectedLanguage === 'ur'
                    ? 'اردو میں متن درج کریں...'
                    : 'Enter text in English...'}
                  rows="6"
                  className={`field-input resize-none ${selectedLanguage === 'ur' ? 'ur-text' : ''}`}
                  dir={selectedLanguage === 'ur' ? 'rtl' : 'ltr'}
                />
                <p className={`mt-2 text-xs text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
                  {text.length} {language === 'en' ? 'characters' : 'حروف'} • {text.split(/\s+/).filter(w => w).length} {language === 'en' ? 'words' : 'الفاظ'}
                </p>
              </div>

              <div className="rounded-2xl border border-[#d5e0f2] bg-[#f2f6ff] p-3">
                <p className={`text-sm text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`} dir={language === 'ur' ? 'rtl' : 'ltr'}>
                  💡 {selectedLanguage === 'ur'
                    ? 'اردو میں متن درج کریں۔ ہر لفظ اپنے اشاراتی اینیمیشن کے ساتھ ظاہر ہوگا۔'
                    : (language === 'en'
                      ? 'Enter text to animate. Each word will be displayed with its sign animation.'
                      : 'متن درج کریں۔ ہر لفظ اپنے سائن اینیمیشن کے ساتھ ظاہر ہوگا۔')}
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={handleTranslate}
                  disabled={!text.trim() || isAnimating}
                  className="btn-primary flex-1 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Sparkles className="h-4 w-4" />
                  {isAnimating
                    ? 'Animating...'
                    : t('text_translate')}
                </button>
                <button
                  onClick={handleClear}
                  className="btn-secondary px-4"
                >
                  <Eraser className="h-4 w-4" />
                  {language === 'en' ? 'Clear' : 'صاف کریں'}
                </button>
              </div>
            </div>

            <div className="section-card p-6">
              <h3 className={`mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                <Languages className="h-4 w-4" />
                {language === 'en' ? 'Quick Words' : 'فوری الفاظ'}
              </h3>
              <div className="stagger-grid grid grid-cols-2 gap-2">
                {(selectedLanguage === 'ur' ? quickWordsUrdu : quickWordsEnglish).map((word) => (
                  <button
                    key={word}
                    onClick={() => handleQuickWord(word)}
                    className={`rounded-xl border border-[#d5e0f2] bg-white px-3 py-2 text-left text-sm font-medium text-[#2d3c54] transition-colors hover:bg-[#eef3ff] ${selectedLanguage === 'en' ? 'capitalize' : 'ur-text'}`}
                  >
                    {word}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2">
            <AvatarPlayer
              text={text}
              language={selectedLanguage}
              isAnimating={isAnimating}
              onAnimationStart={handleAnimationStart}
              onAnimationEnd={handleAnimationEnd}
            />

            <div className="section-card mt-6 space-y-2 p-4">
              <h4 className={`font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`} dir={language === 'ur' ? 'rtl' : 'ltr'}>
                💡 {language === 'en' ? 'Tips for Best Results' : 'بہترین نتائج کے لیے تجاویز'}
              </h4>
              <ul className={`list-inside list-disc space-y-1 text-sm text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`} dir={language === 'ur' ? 'rtl' : 'ltr'}>
                <li>{selectedLanguage === 'ur'
                  ? 'عام الفاظ استعمال کریں جو فہرست میں موجود ہیں'
                  : (language === 'en'
                    ? 'Use simple words from the available list'
                    : 'دستیاب فہرست سے سادہ الفاظ استعمال کریں')}</li>
                <li>{selectedLanguage === 'ur'
                  ? 'ایک وقت میں ایک لفظ بہتر کام کرتا ہے'
                  : (language === 'en'
                    ? 'One word at a time works best'
                    : 'ایک وقت میں ایک لفظ بہتر ہے')}</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="section-card p-5">
          <p className={`mb-2 text-sm font-semibold text-[#3c6fe2] ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en'
              ? '✅ 32 Sign Language Animations Available'
              : '✅ 32 اشاراتی زبان کی اینیمیشنز دستیاب ہیں'}
          </p>
          {selectedLanguage === 'ur' ? (
            <p className="ur-text text-sm text-[#2d3c54]" dir="rtl">
              یہ الفاظ آزمائیں: الرٹ، کتاب، محتاط، سستا، پاگل، خطرناک، مہذب، احمق، خوش، انتہائی، بہترین، دور، خوفزدہ، غیر ملکی، مضحکہ خیز، اچھا، صحت مند، بھاری، اہم، عقلمند، دلچسپ، دیر، کم، نیا، نہیں، شور، پرامن، تیز، تیار، محفوظ، ذہین، ہاں
            </p>
          ) : (
            <p className="text-sm text-[#2d3c54]">
              Try these words: alert, book, careful, cheap, crazy, dangerous, decent, dumb, excited, extreme, fantastic, far, fearful, foreign, funny, good, healthy, heavy, important, intelligent, interesting, late, less, new, no, noisy, peaceful, quick, ready, secure, smart, yes
            </p>
          )}
          <div className="mt-3 flex items-center gap-2 text-xs text-[#60708b]">
            <ChevronRight className="h-4 w-4" />
            {language === 'en' ? 'Click any quick word to preview instantly.' : 'فوری پیش نظارہ کے لیے کسی لفظ پر کلک کریں۔'}
          </div>
        </div>
      </div>
    </div>
  );
};
