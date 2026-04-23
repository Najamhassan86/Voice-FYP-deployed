import React, { useEffect, useState, useMemo } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Play, Pause } from 'lucide-react';
import { AnimationPlayer3D } from './AnimationPlayer3D';
import { resolveAnimationForPhrase } from '../api/animationApi';

export const AvatarPlayer = ({ text, language, isAnimating, onAnimationStart, onAnimationEnd }) => {
  const { t } = useLanguage();
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentAnimation, setCurrentAnimation] = useState(null);
  const [loadingAnimation, setLoadingAnimation] = useState(false);

  // Memoize words array to prevent infinite re-renders
  const words = useMemo(() => {
    return text ? text.split(' ').filter(w => w.length > 0) : [];
  }, [text]);

  // Load default animation on mount (show model immediately)
  useEffect(() => {
    const loadDefaultAnimation = async () => {
      try {
        const result = await resolveAnimationForPhrase('ready', 'psl');
        setCurrentAnimation(result.animation);
      } catch (error) {
        console.error('Failed to load default animation:', error);
      }
    };

    if (!text && !currentAnimation) {
      loadDefaultAnimation();
    }
  }, []);

  // Reset state when text changes (handles interruption)
  useEffect(() => {
    setCurrentWordIndex(0);
    if (text) {
      setCurrentAnimation(null);
    }
  }, [text]);

  // Load animation for current word (only when word index or text changes)
  useEffect(() => {
    if (words.length > 0 && currentWordIndex < words.length && !loadingAnimation) {
      const currentWord = words[currentWordIndex];
      loadAnimationForWord(currentWord);
    }
  }, [currentWordIndex, text]); // Depend on text, not words array

  // Auto-play when isAnimating prop changes
  useEffect(() => {
    if (isAnimating && words.length > 0) {
      handlePlay();
    } else if (!isAnimating) {
      setIsPlaying(false);
    }
  }, [isAnimating]);

  const loadAnimationForWord = async (word) => {
    setLoadingAnimation(true);
    try {
      const result = await resolveAnimationForPhrase(word, language);
      setCurrentAnimation(result.animation);
      console.log(`Loaded animation for "${word}":`, result.animation);
    } catch (error) {
      console.error(`Failed to load animation for "${word}":`, error);
      setCurrentAnimation(null);
    } finally {
      setLoadingAnimation(false);
    }
  };

  const handlePlay = () => {
    setCurrentWordIndex(0);
    setIsPlaying(true);
    onAnimationStart?.();
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleAnimationComplete = () => {
    console.log(`Animation complete for word ${currentWordIndex + 1}/${words.length}`);

    // Move to next word after animation completes
    setTimeout(() => {
      if (currentWordIndex < words.length - 1) {
        setCurrentWordIndex(prev => prev + 1);
      } else {
        // All words complete
        setIsPlaying(false);
        onAnimationEnd?.();
        setCurrentWordIndex(0);
      }
    }, 500); // Small delay between words
  };

  const currentWord = words[currentWordIndex] || '';

  return (
    <div className="section-card space-y-6 p-6 sm:p-8">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="eyebrow mb-1 text-[#5f73a5]">Avatar Rendering</p>
          <h3 className="text-xl font-semibold text-[#1a2642]">3D PSL Animation Player</h3>
        </div>
        {words.length > 0 && (
          <span className="rounded-full border border-[#d5e0f2] bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wide text-[#60708b]">
            {currentWordIndex + 1}/{words.length}
          </span>
        )}
      </div>

      <div className="relative aspect-video overflow-hidden rounded-2xl border border-[#c8d8f2] bg-[#eef3ff]">
        {currentAnimation ? (
          <>
            <AnimationPlayer3D
              key={`${currentAnimation.id}-${currentWordIndex}`}
              animationPath={currentAnimation.file_path}
              isPlaying={isPlaying}
              animationSpeed={1.0}
              onAnimationComplete={handleAnimationComplete}
            />
            {text && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 transform">
                <div className="rounded-full bg-[#4f46e5]/95 px-6 py-2 text-lg font-semibold text-white shadow-xl">
                  {currentWord}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex h-full items-center justify-center bg-gradient-to-br from-[#f4f7ff] to-[#e8f1ff]">
            <div className="text-center text-[#60708b]">
              <p className="text-sm mt-2">
                {loadingAnimation ? 'Loading 3D model...' : 'Preparing avatar...'}
              </p>
            </div>
          </div>
        )}
      </div>

      {words.length > 0 && (
        <div className="text-center text-sm text-[#60708b]">
          <p>
            {currentWordIndex + 1} / {words.length} words
          </p>
          <div className="mt-3 h-2 w-full rounded-full bg-[#dbe6f8]">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] transition-all duration-300"
              style={{
                width: `${((currentWordIndex + 1) / words.length) * 100}%`
              }}
            />
          </div>
        </div>
      )}

      <div className="flex gap-3">
        {!isPlaying ? (
          <button
            onClick={handlePlay}
            disabled={!text || loadingAnimation}
            className="btn-primary flex-1 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Play className="w-5 h-5" />
            {t('text_replay')}
          </button>
        ) : (
          <button
            onClick={handlePause}
            className="flex flex-1 items-center justify-center gap-2 rounded-full bg-gradient-to-r from-[#ff8a3d] to-[#ff5d5d] px-4 py-3 font-medium text-white"
          >
            <Pause className="w-5 h-5" />
            Pause
          </button>
        )}
      </div>

      {words.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold uppercase tracking-wide text-[#5f73a5]">
            Words to animate
          </h4>
          <div className="flex flex-wrap gap-2">
            {words.map((word, idx) => (
              <span
                key={idx}
                className={`px-3 py-1 rounded-full text-sm font-medium transition-all ${
                  idx === currentWordIndex
                    ? 'scale-110 bg-[#4f46e5] text-white'
                    : idx < currentWordIndex
                    ? 'bg-[#d8f4ea] text-[#117249]'
                    : 'bg-[#e9efff] text-[#4c5d81]'
                }`}
              >
                {word}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
