import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Globe, Sparkles } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

export const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { language, setLanguage, t } = useLanguage();
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  const navLinks = [
    { path: '/', label: t('nav_home') },
    { path: '/psl-to-text', label: t('nav_psl_to_text') },
    { path: '/text-to-psl', label: t('nav_text_to_psl') },
    { path: '/learn', label: t('nav_learn') },
  ];

  return (
    <nav className="sticky top-0 z-50 px-2 pt-3 sm:px-3">
      <div className="page-wrap">
        <div className="relative overflow-hidden rounded-[22px] border border-[#cad8f1]/90 bg-[#fdfefe]/82 shadow-[0_18px_50px_rgba(35,48,91,0.22)] backdrop-blur-2xl">
          <div className="pointer-events-none absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-[#7a7dff] to-transparent" />

          <div className="flex h-20 items-center justify-between gap-3 px-3 sm:px-5">
            <Link to="/" className="group flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] text-white shadow-lg shadow-indigo-900/30 transition-transform duration-300 group-hover:scale-[1.06]">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-[#5f73a5]">Pakistani Sign Tech</p>
                <p className="text-lg font-semibold leading-none text-[#1a2642]">VOICE</p>
              </div>
            </Link>

            <div className="stagger-grid hidden items-center gap-1 rounded-full border border-[#cad8f1] bg-white/88 p-1.5 md:flex">
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition-all duration-300 ${
                    isActive(link.path)
                      ? 'bg-gradient-to-r from-[#4f46e5] to-[#2ea6ff] text-white shadow-[0_8px_24px_rgba(79,70,229,0.33)]'
                      : 'text-[#2d3c54] hover:bg-[#edf3ff] hover:text-[#1c2f4c]'
                  } ${language === 'ur' ? 'ur-text' : ''}`}
                >
                  <span className="inline-flex items-center gap-2">
                    {isActive(link.path) && <span className="h-1.5 w-1.5 rounded-full bg-white/95" />}
                    {link.label}
                  </span>
                </Link>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setLanguage(language === 'en' ? 'ur' : 'en')}
                className="btn-secondary border-[#c7d6f0] bg-white px-3 py-2 shadow-sm"
                title={t('language_select')}
              >
                <Globe className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wider">
                  {language === 'en' ? 'EN' : 'UR'}
                </span>
              </button>

              <button
                onClick={() => setIsOpen(!isOpen)}
                className="rounded-xl border border-[#cbd8f0] bg-white/95 p-2.5 text-[#2d3c54] shadow-sm md:hidden"
                aria-label="Toggle menu"
              >
                {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              </button>
            </div>
          </div>
        </div>

        {isOpen && (
          <div className="stagger-grid mt-3 animate-slide-in rounded-2xl border border-[#cad8f1] bg-white p-3 shadow-[0_16px_38px_rgba(35,48,91,0.2)] md:hidden">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setIsOpen(false)}
                className={`mb-1 block rounded-xl px-4 py-2.5 text-sm font-medium last:mb-0 ${
                  isActive(link.path)
                    ? 'bg-gradient-to-r from-[#4f46e5] to-[#2ea6ff] text-white shadow-[0_8px_22px_rgba(79,70,229,0.28)]'
                    : 'text-[#2d3c54] hover:bg-[#edf3ff]'
                } ${language === 'ur' ? 'ur-text' : ''}`}
              >
                {link.label}
              </Link>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
};
