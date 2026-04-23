import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { usePSLRecognition } from '../hooks/usePSLRecognition';
import { getPSLModelInfo, getActivePSLBaseURL } from '../api/pslApi';
import { Video, VideoOff, RotateCcw, Hand } from 'lucide-react';

const PROGRESS_STORAGE_KEY = 'psl_learn_progress_v1';
const LEARN_FEEDBACK_MIN_CONFIDENCE = 0.35;

const loadProgress = () => {
  try {
    if (typeof window === 'undefined' || !window.localStorage) {
      return {
        attempts: 0,
        correct: 0,
        sessions: 0,
        practicedWords: []
      };
    }

    const raw = window.localStorage.getItem(PROGRESS_STORAGE_KEY);
    if (!raw) {
      return {
        attempts: 0,
        correct: 0,
        sessions: 0,
        practicedWords: []
      };
    }
    const parsed = JSON.parse(raw);
    return {
      attempts: Number(parsed.attempts || 0),
      correct: Number(parsed.correct || 0),
      sessions: Number(parsed.sessions || 0),
      practicedWords: Array.isArray(parsed.practicedWords) ? parsed.practicedWords : []
    };
  } catch {
    return {
      attempts: 0,
      correct: 0,
      sessions: 0,
      practicedWords: []
    };
  }
};

export const Learn = () => {
  const { t, language } = useLanguage();
  const videoRef = useRef(null);
  const [modelClasses, setModelClasses] = useState([]);
  const [selectedWord, setSelectedWord] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [progress, setProgress] = useState(loadProgress);
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [modelInfoError, setModelInfoError] = useState('');
  const [activeApiBaseURL, setActiveApiBaseURL] = useState('');
  const [lastFeedbackAt, setLastFeedbackAt] = useState(null);
  const handsDetectedRef = useRef(0);

  const recognition = usePSLRecognition({
    sequenceLength: 60,
    confidenceThreshold: 0.6,
    cooldownMs: 1500,
    autoRecognize: true,
    allowedLabels: modelClasses
  });

  const mediaPipe = useMediaPipe(videoRef, {
    onResults: (results) => {
      handsDetectedRef.current = results?.multiHandLandmarks?.length || 0;
    },
    onFeatures: (features) => {
      if (handsDetectedRef.current <= 0) {
        return;
      }
      recognition.addFrame(features);
    },
    drawLandmarks: true
  });

  const fetchModelInfo = useCallback(async () => {
    setIsLoadingModel(true);
    setModelInfoError('');
    try {
      const info = await getPSLModelInfo();
      setActiveApiBaseURL(getActivePSLBaseURL() || '');
      const classes = Array.isArray(info?.classes) ? info.classes : [];
      setModelClasses(classes);
      if (classes.length > 0) {
        setSelectedWord((prev) => (prev && classes.includes(prev) ? prev : classes[0]));
      }
      if (classes.length === 0) {
        setModelInfoError(language === 'en' ? 'No classes returned by backend model info.' : 'بیک اینڈ سے کلاسز موصول نہیں ہوئیں۔');
      }
    } catch (err) {
      setModelClasses([]);
      setModelInfoError(
        language === 'en'
          ? `Could not connect to backend: ${err?.message || 'Unknown error'}`
          : `بیک اینڈ سے رابطہ نہ ہو سکا: ${err?.message || 'نامعلوم خرابی'}`
      );
    } finally {
      setIsLoadingModel(false);
    }
  }, [language]);

  useEffect(() => {
    fetchModelInfo();
  }, [fetchModelInfo]);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.localStorage) {
      return;
    }
    window.localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(progress));
  }, [progress]);

  const similarity = feedback?.similarity || 0;

  const hints = useMemo(() => {
    const baseHints = [
      language === 'en' ? 'Keep your hand fully visible inside the camera frame.' : 'اپنا ہاتھ کیمرے کے فریم میں مکمل طور پر رکھیں۔',
      language === 'en' ? 'Hold the sign steady for about 2 seconds.' : 'اشارے کو تقریباً 2 سیکنڈ تک مستحکم رکھیں۔',
      language === 'en' ? 'Face your palm similarly to the reference sign shape.' : 'اپنی ہتھیلی کی سمت حوالہ اشارے کے مطابق رکھیں۔'
    ];

    if (!feedback?.predictedWord || feedback.isCorrect) {
      return baseHints;
    }

    return [
      language === 'en'
        ? `The model read your sign as "${feedback.predictedWord}". Slow down and exaggerate the shape for "${selectedWord}".`
        : `ماڈل نے آپ کا اشارہ "${feedback.predictedWord}" سمجھا۔ "${selectedWord}" کے لیے حرکت کو آہستہ اور واضح کریں۔`,
      ...baseHints.slice(0, 2)
    ];
  }, [language, feedback, selectedWord]);

  const handleStartPractice = useCallback(async () => {
    if (!selectedWord || modelClasses.length === 0) {
      setModelInfoError(
        language === 'en'
          ? 'Select a target word first after model sync.'
          : 'ماڈل ہم آہنگی کے بعد پہلے ہدف لفظ منتخب کریں۔'
      );
      return;
    }

    setFeedback(null);
    recognition.resetBuffer();
    recognition.clearSentence();
    try {
      await mediaPipe.startDetection();
      setProgress((prev) => ({ ...prev, sessions: prev.sessions + 1 }));
    } catch {
      // Error is exposed via mediaPipe.error
    }
  }, [language, mediaPipe, modelClasses.length, recognition, selectedWord]);

  const handleStopPractice = useCallback(() => {
    mediaPipe.stopDetection();
    recognition.resetBuffer();
  }, [mediaPipe, recognition]);

  const handleTryAgain = useCallback(() => {
    setFeedback(null);
    recognition.resetBuffer();
    recognition.clearSentence();
  }, [recognition]);

  const handleResetProgress = useCallback(() => {
    const empty = {
      attempts: 0,
      correct: 0,
      sessions: 0,
      practicedWords: []
    };
    setProgress(empty);
    setFeedback(null);
  }, []);

  useEffect(() => {
    const prediction = recognition.currentPrediction;
    if (!prediction || !selectedWord) {
      return;
    }

    const usableConfidence = prediction.confidence >= LEARN_FEEDBACK_MIN_CONFIDENCE;
    if (!usableConfidence) {
      return;
    }

    const normalizedLabel = String(prediction.label || '').toLowerCase();
    const normalizedSelected = String(selectedWord).toLowerCase();
    const isCorrect = normalizedLabel === normalizedSelected;
    const computedSimilarity = Math.max(35, Math.min(99, Math.round(prediction.confidence * 100)));

    setFeedback({
      isCorrect,
      predictedWord: prediction.label,
      similarity: computedSimilarity,
      confidence: prediction.confidence
    });
    setLastFeedbackAt(Date.now());

    setProgress((prev) => {
      const nextPracticed = prev.practicedWords.includes(selectedWord)
        ? prev.practicedWords
        : [...prev.practicedWords, selectedWord];

      return {
        ...prev,
        attempts: prev.attempts + 1,
        correct: prev.correct + (isCorrect ? 1 : 0),
        practicedWords: nextPracticed
      };
    });
  }, [recognition.currentPrediction, selectedWord]);

  const averageScore = progress.attempts > 0
    ? Math.round((progress.correct / progress.attempts) * 100)
    : 0;

  return (
    <div className="pt-8">
      <div className="page-wrap space-y-8 reveal-up">
        <div className="section-card p-8 sm:p-10">
          <p className="eyebrow mb-3 text-[#5f73a5]">Learning Studio</p>
          <h1 className={`title-display mb-2 text-4xl ${language === 'ur' ? 'ur-text' : ''}`}>
            {t('learn_title')}
          </h1>
          <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
            {t('learn_subtitle')}
          </p>
          {activeApiBaseURL && (
            <p className={`mt-2 text-xs text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en' ? `Connected backend: ${activeApiBaseURL}` : `منسلک بیک اینڈ: ${activeApiBaseURL}`}
            </p>
          )}
        </div>

        <div className="stagger-grid grid gap-8 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            <div className="overflow-hidden rounded-2xl border border-[#c8d8f2] bg-[#0b1324] shadow-lg">
              <div className="relative w-full bg-black aspect-video flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 ${
                    mediaPipe.isDetecting ? 'opacity-100' : 'opacity-0'
                  }`}
                />

                <canvas
                  ref={mediaPipe.canvasRef}
                  className={`absolute inset-0 w-full h-full transform -scale-x-100 ${
                    mediaPipe.isDetecting ? 'opacity-100' : 'opacity-0'
                  }`}
                />

                {mediaPipe.isDetecting ? (
                  <>
                    <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/55 px-3 py-2 rounded-lg text-white text-sm font-medium">
                      <Hand className={`w-4 h-4 ${mediaPipe.handsDetected > 0 ? 'text-green-400' : 'text-gray-300'}`} />
                      {mediaPipe.handsDetected} {language === 'en' ? 'hand(s)' : 'ہاتھ'}
                    </div>

                    <div className="absolute top-4 right-4 rounded-lg bg-black/55 px-3 py-2 text-white text-sm font-medium">
                      {mediaPipe.fps} FPS
                    </div>

                    <div className="absolute bottom-4 left-4 right-4 rounded-lg bg-black/55 p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white text-xs font-medium">
                          {language === 'en' ? 'Learning buffer' : 'لرننگ بفر'}: {recognition.bufferLength}/60
                        </span>
                        <span className="text-white text-xs">
                          {recognition.bufferProgress.toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-[#243049]">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] transition-all duration-300"
                          style={{ width: `${recognition.bufferProgress}%` }}
                        />
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center">
                    <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-[#4f46e5]/25">
                      <svg
                        className="h-8 w-8 text-[#7bb6ff]"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path d="M4 4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4z" />
                      </svg>
                    </div>
                    <p className="text-white text-lg font-medium">
                      {language === 'en' ? 'Ready to practice?' : 'سیکھنے کے لیے تیار؟'}
                    </p>
                  </div>
                )}
              </div>

              <div className="flex gap-4 bg-[#f9fbff] px-6 py-4">
                {!mediaPipe.isDetecting ? (
                  <button
                    onClick={handleStartPractice}
                    disabled={isLoadingModel || modelClasses.length === 0 || !selectedWord}
                    className="btn-primary flex-1 disabled:opacity-50"
                  >
                    <Video className="w-5 h-5" />
                    {language === 'en' ? 'Start Practice' : 'مشق شروع کریں'}
                  </button>
                ) : (
                  <button
                    onClick={handleStopPractice}
                    className="btn-danger flex-1"
                  >
                    <VideoOff className="w-5 h-5" />
                    {language === 'en' ? 'Stop Practice' : 'مشق بند کریں'}
                  </button>
                )}

                <button
                  onClick={handleTryAgain}
                  className="btn-secondary"
                  disabled={!mediaPipe.isDetecting && !feedback}
                >
                  <RotateCcw className="w-5 h-5" />
                  {language === 'en' ? 'Reset Attempt' : 'کوشش دوبارہ کریں'}
                </button>
              </div>
            </div>

            {(feedback || recognition.currentPrediction) && (
              <div className="section-card space-y-4 p-6">
                <h3 className={`text-lg font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                  {language === 'en' ? 'Feedback & Comparison' : 'رائے اور موازنہ'}
                </h3>

                <div className="stagger-grid grid grid-cols-2 gap-4">
                  <div className="rounded-2xl bg-[#eef3ff] p-4 text-center">
                    <p className={`mb-3 text-xs font-semibold uppercase text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {t('learn_reference')}
                    </p>
                    <div className="text-5xl mb-3">🎯</div>
                    <p className="font-semibold text-[#1a2642]">{selectedWord || '-'}</p>
                  </div>

                  <div className="rounded-2xl border border-[#ffd7be] bg-[#fff2e8] p-4 text-center">
                    <p className={`mb-3 text-xs font-semibold uppercase text-[#8f6d56] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {t('learn_your_attempt')}
                    </p>
                    <div className="text-5xl mb-3">✋</div>
                    <p className="font-semibold text-[#1a2642]">
                      {recognition.currentPrediction?.label || (language === 'en' ? 'Waiting...' : 'انتظار...')}
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl border border-[#d5e0f2] bg-white p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className={`font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {t('learn_similarity')}
                    </h4>
                    <span className="text-3xl font-bold text-[#4f46e5]">
                      {similarity}%
                    </span>
                  </div>
                  <div className="h-3 w-full overflow-hidden rounded-full bg-[#dbe6f8]">
                    <div
                      className="h-3 rounded-full bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] transition-all duration-300"
                      style={{ width: `${similarity}%` }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-[#60708b]">
                    {feedback?.isCorrect
                      ? (language === 'en' ? `Great! You matched "${selectedWord}".` : `بہترین! آپ نے "${selectedWord}" درست کیا۔`)
                      : (language === 'en'
                        ? 'Not a perfect match yet. Use hints below and try again.'
                        : 'ابھی مکمل مماثلت نہیں۔ نیچے دی گئی ہدایات پر عمل کریں اور دوبارہ کوشش کریں۔')}
                  </p>
                  {lastFeedbackAt && (
                    <p className="mt-1 text-xs text-[#7b8798]">
                      {language === 'en'
                        ? `Last update: ${new Date(lastFeedbackAt).toLocaleTimeString()}`
                        : `آخری تازہ کاری: ${new Date(lastFeedbackAt).toLocaleTimeString()}`}
                    </p>
                  )}
                </div>

                <div className="rounded-2xl border border-[#d5e0f2] bg-[#f6f9ff] p-4">
                  <h4 className={`mb-3 font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {t('learn_hints')}
                  </h4>
                  <ul className="space-y-2">
                    {hints.map((suggestion, idx) => (
                      <li key={idx} className={`flex items-start gap-2 text-sm text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`}>
                        <span className="mt-0.5 inline-flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-[#4f46e5] text-xs text-white">
                          {idx + 1}
                        </span>
                        <span>{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={handleTryAgain}
                    className="btn-primary flex-1"
                  >
                    <RotateCcw className="w-5 h-5" />
                    {language === 'en' ? 'Try Again' : 'دوبارہ کوشش کریں'}
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="section-card p-6">
              <h3 className={`mb-4 text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                {language === 'en' ? 'Practice Words' : 'سیکھنے کے الفاظ'}
              </h3>
              <div className="space-y-3 max-h-[420px] overflow-auto pr-1">
                {modelClasses.map((word) => (
                  <div
                    key={word}
                    onClick={() => {
                      setSelectedWord(word);
                      setFeedback(null);
                      recognition.clearSentence();
                      recognition.resetBuffer();
                    }}
                    className={`cursor-pointer rounded-2xl border p-4 transition-colors ${
                      selectedWord === word
                        ? 'border-[#4f46e5] bg-[#eef3ff]'
                        : 'border-[#d5e0f2] bg-white hover:bg-[#eef3ff]'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🖐️</span>
                      <span className="font-semibold text-[#1a2642]">
                        {word}
                      </span>
                    </div>
                  </div>
                ))}

                {!isLoadingModel && modelClasses.length === 0 && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                    <p className={`text-sm text-amber-800 ${language === 'ur' ? 'ur-text' : ''}`}>
                      {modelInfoError || (language === 'en'
                        ? 'No model classes found. Start backend and retry.'
                        : 'ماڈل کلاسز نہیں ملیں۔ بیک اینڈ شروع کریں اور دوبارہ کوشش کریں۔')}
                    </p>
                    <button
                      onClick={fetchModelInfo}
                      className="btn-secondary mt-2"
                    >
                      {language === 'en' ? 'Retry Model Sync' : 'ماڈل دوبارہ حاصل کریں'}
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="section-card p-6">
              <h3 className={`mb-4 text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                {language === 'en' ? 'Your Progress' : 'آپ کی ترقی'}
              </h3>
              <div className="space-y-4">
                <div>
                  <p className={`mb-1 text-sm text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Signs Practiced' : 'سیکھے گئے اشارے'}
                  </p>
                  <p className="text-3xl font-bold text-[#4f46e5]">{progress.practicedWords.length}</p>
                </div>
                <div>
                  <p className={`mb-1 text-sm text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Average Score' : 'اوسط اسکور'}
                  </p>
                  <p className="text-3xl font-bold text-[#4f46e5]">{averageScore}%</p>
                </div>
                <div>
                  <p className={`mb-1 text-sm text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Total Sessions' : 'کل سیشنز'}
                  </p>
                  <p className="text-3xl font-bold text-[#4f46e5]">{progress.sessions}</p>
                </div>
              </div>

              <button
                onClick={handleResetProgress}
                className="btn-secondary mt-4 w-full"
              >
                {language === 'en' ? 'Reset Progress' : 'پیش رفت دوبارہ ترتیب دیں'}
              </button>
            </div>
          </div>
        </div>

        <div className="section-card p-4">
          <p className={`mb-2 text-sm font-semibold text-[#3c6fe2] ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en' ? 'Live Learning Mode Powered by PSL Recognition' : 'براہ راست لرننگ موڈ PSL شناخت کے ساتھ'}
          </p>
          <ul className={`list-inside list-disc space-y-1 text-sm text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`}>
            <li>{language === 'en' ? 'Select a target word from the model class list.' : 'ماڈل کلاس فہرست میں سے ہدف لفظ منتخب کریں۔'}</li>
            <li>{language === 'en' ? 'Start practice and perform the sign in front of the camera.' : 'مشق شروع کریں اور کیمرے کے سامنے اشارہ کریں۔'}</li>
            <li>{language === 'en' ? 'You receive instant match feedback and confidence-based similarity.' : 'آپ کو فوری مماثلت اور اعتماد پر مبنی فیڈبیک ملتا ہے۔'}</li>
          </ul>

          {(mediaPipe.error || recognition.error) && (
            <p className="mt-3 text-sm font-semibold text-red-700">
              {language === 'en' ? 'Error:' : 'خرابی:'} {mediaPipe.error || recognition.error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
