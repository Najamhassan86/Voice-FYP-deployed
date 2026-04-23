import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { ExternalLink, Phone, ShieldCheck } from 'lucide-react';

export const Footer = () => {
  const { language } = useLanguage();

  return (
    <footer className="mt-16 border-t border-[#cad8f1] bg-gradient-to-b from-[#eef3ff]/85 to-[#f8fbff]/85">
      <div className="page-wrap py-12">
        <div className="section-card grid gap-8 p-8 md:grid-cols-3">
          <div>
            <p className="eyebrow mb-3">Voice Platform</p>
            <h3 className="mb-3 text-2xl font-semibold text-[#1a2642]">Built For Inclusive Communication</h3>
            <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en'
                ? 'Real-time translation between Pakistani Sign Language, text, and speech for practical, everyday use.'
                : 'پاکستانی سائن لینگویج، متن اور آواز کے درمیان حقیقی وقت میں ترجمہ۔'}
            </p>
          </div>

          <div>
            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.14em] text-[#5c6d64]">
              {language === 'en' ? 'Resources' : 'وسائل'}
            </p>
            <div className="space-y-3 text-sm text-[#2d3c54]">
              <a className="flex items-center gap-2 hover:text-[#4f46e5]" href="https://github.com/Najamhassan86/VOICE-FYP" target="_blank" rel="noopener noreferrer">
                GitHub Repository <ExternalLink className="h-4 w-4" />
              </a>
              <a className="flex items-center gap-2 hover:text-[#4f46e5]" href="https://github.com/Najamhassan86/VOICE-FYP/blob/main/README.md" target="_blank" rel="noopener noreferrer">
                Documentation <ExternalLink className="h-4 w-4" />
              </a>
              <a className="flex items-center gap-2 hover:text-[#4f46e5]" href="https://github.com/Najamhassan86/VOICE-FYP/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
                {language === 'en' ? 'License' : 'لائسنس'} <ShieldCheck className="h-4 w-4" />
              </a>
              <a className="flex items-center gap-2 hover:text-[#4f46e5]" href="tel:+923171975525">
                <Phone className="h-4 w-4" /> 0317 1975525
              </a>
            </div>
          </div>

          <div>
            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.14em] text-[#5c6d64]">
              {language === 'en' ? 'Credits' : 'کریڈٹس'}
            </p>
            <p className="text-sm text-[#2d3c54]">FAST National University of Computer and Emerging Sciences</p>
            <p className={`mt-2 text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en' ? 'Supervisor: Mr. Almas Khan' : 'نگرانی کے تحت: Mr. Almas Khan'}
            </p>
            <p className="mt-2 text-sm text-[#5c6d64]">Final Year Project 2022-2026</p>
          </div>
        </div>

        <div className="pt-6 text-center text-xs tracking-wide text-[#5c6d64]">
          <p>
            © 2026 VOICE. {language === 'en' ? 'All rights reserved.' : 'تمام حقوق محفوظ ہیں۔'}
          </p>
        </div>
      </div>
    </footer>
  );
};
