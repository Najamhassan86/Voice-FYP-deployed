import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { usePSLRecognition } from '../hooks/usePSLRecognition';
import { getPSLModelInfo, getActivePSLBaseURL, scorePracticePSL, recognizePSL, validateSequence } from '../api/pslApi';
import { resolveAnimationForPhrase } from '../api/animationApi';
import { normalizePSLClasses } from '../constants/pslClasses';
import { AnimationPlayer3D } from '../components/AnimationPlayer3D';
import { VideoOff, RotateCcw, Hand, Play, ArrowLeft, GraduationCap, ClipboardCheck } from 'lucide-react';

const PROGRESS_STORAGE_KEY = 'psl_learn_progress_v1';
const SCORE_CORRECT_THRESHOLD = 72;
/** Prep time after "Check my sign" before any frames are recorded */
const CAPTURE_PREP_MS = 2000;
/** Max frames to record (~4.5s at 30fps) so longer signs fit in one attempt */
const MAX_CAPTURE_FRAMES = 60;
/** Rolling window size (model input) */
const WINDOW_FRAMES = 60;
/** Consecutive polls where classifier matches target word above threshold */
const STABLE_MATCH_STREAK = 2;
const STABLE_MATCH_CONF = 0.48;
const LISTEN_POLL_MS = 320;

const loadProgress = () => {
  try {
    if (typeof window === 'undefined' || !window.localStorage) {
      return { attempts: 0, correct: 0, sessions: 0, practicedWords: [] };
    }
    const raw = window.localStorage.getItem(PROGRESS_STORAGE_KEY);
    if (!raw) {
      return { attempts: 0, correct: 0, sessions: 0, practicedWords: [] };
    }
    const parsed = JSON.parse(raw);
    return {
      attempts: Number(parsed.attempts || 0),
      correct: Number(parsed.correct || 0),
      sessions: Number(parsed.sessions || 0),
      practicedWords: Array.isArray(parsed.practicedWords) ? parsed.practicedWords : []
    };
  } catch {
    return { attempts: 0, correct: 0, sessions: 0, practicedWords: [] };
  }
};

export const Learn = () => {
  const { t, language } = useLanguage();
  const videoRef = useRef(null);
  const handsDetectedRef = useRef(0);
  const learnPhaseRef = useRef('pick');
  const signCapturePhaseRef = useRef('idle');
  /** When true, frames are appended to the extended capture ring */
  const ingestCaptureRef = useRef(false);
  /** Full capture session (188-d vectors), max MAX_CAPTURE_FRAMES */
  const captureRingRef = useRef([]);
  const stableHitsRef = useRef(0);
  const pollingBusyRef = useRef(false);
  const captureSessionRef = useRef(0);

  const [modelClasses, setModelClasses] = useState([]);
  const [selectedWord, setSelectedWord] = useState('');
  const [learnPhase, setLearnPhase] = useState('pick');
  const [progress, setProgress] = useState(loadProgress);
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [modelInfoError, setModelInfoError] = useState('');
  const [activeApiBaseURL, setActiveApiBaseURL] = useState('');

  const [currentAnimation, setCurrentAnimation] = useState(null);
  const [loadingAnim, setLoadingAnim] = useState(false);
  const [animError, setAnimError] = useState('');
  const [isAnimPlaying, setIsAnimPlaying] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [modelScale, setModelScale] = useState(1.5);

  const [practiceResult, setPracticeResult] = useState(null);
  const [modelGuess, setModelGuess] = useState(null);
  const [scoreError, setScoreError] = useState('');
  const [isScoring, setIsScoring] = useState(false);
  /** idle | countdown | listening | finishing | done */
  const [signCapturePhase, setSignCapturePhase] = useState('idle');
  const [prepMsLeft, setPrepMsLeft] = useState(0);
  const [captureFrameCount, setCaptureFrameCount] = useState(0);
  const [listenStatus, setListenStatus] = useState('');
  const [captureSessionTick, setCaptureSessionTick] = useState(0);

  const recognition = usePSLRecognition({
    sequenceLength: 60,
    confidenceThreshold: 0.5,
    cooldownMs: 800,
    autoRecognize: false,
    allowedLabels: modelClasses,
    getHandsDetected: () => handsDetectedRef.current
  });

  useEffect(() => {
    learnPhaseRef.current = learnPhase;
  }, [learnPhase]);

  useEffect(() => {
    signCapturePhaseRef.current = signCapturePhase;
  }, [signCapturePhase]);

  const stopSignCapture = useCallback(() => {
    captureSessionRef.current += 1;
    ingestCaptureRef.current = false;
    stableHitsRef.current = 0;
    captureRingRef.current = [];
    signCapturePhaseRef.current = 'idle';
    setCaptureFrameCount(0);
    setPrepMsLeft(0);
    setListenStatus('');
    setSignCapturePhase('idle');
    recognition.resetBuffer();
  }, [recognition]);

  const mediaPipe = useMediaPipe(videoRef, {
    onResults: (results) => {
      handsDetectedRef.current = results?.multiHandLandmarks?.length || 0;
    },
    onFeatures: (features) => {
      if (learnPhaseRef.current !== 'test') return;
      if (signCapturePhaseRef.current === 'listening' && ingestCaptureRef.current && handsDetectedRef.current > 0) {
        const ring = captureRingRef.current;
        const frame = features.slice();
        ring.push(frame);
        if (ring.length > MAX_CAPTURE_FRAMES) {
          ring.shift();
        }
        if (ring.length % 4 === 0 || ring.length === WINDOW_FRAMES) {
          setCaptureFrameCount(ring.length);
        }
        return;
      }
      if (['idle', 'done'].includes(signCapturePhaseRef.current) && handsDetectedRef.current > 0) {
        recognition.addFrame(features);
      }
    },
    drawLandmarks: true
  });

  const fetchModelInfo = useCallback(async () => {
    setIsLoadingModel(true);
    setModelInfoError('');
    try {
      const info = await getPSLModelInfo();
      setActiveApiBaseURL(getActivePSLBaseURL() || '');
      const classes = normalizePSLClasses(info);
      setModelClasses(classes);
      if (classes.length > 0) {
        setSelectedWord((prev) => (prev && classes.includes(prev) ? prev : classes[0]));
      }
      if (classes.length === 0) {
        setModelInfoError(
          info?.error ||
          (language === 'en'
            ? 'No classes returned by backend model info.'
            : 'بیک اینڈ سے کلاسز موصول نہیں ہوئیں۔')
        );
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
    if (typeof window === 'undefined' || !window.localStorage) return;
    window.localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(progress));
  }, [progress]);

  useEffect(() => {
    if (learnPhase !== 'watch' || !selectedWord) return undefined;
    let cancelled = false;
    setAnimError('');
    setLoadingAnim(true);
    setCurrentAnimation(null);
    (async () => {
      try {
        const result = await resolveAnimationForPhrase(selectedWord, 'psl');
        if (!cancelled) {
          setCurrentAnimation(result.animation);
        }
      } catch (e) {
        if (!cancelled) {
          setAnimError(e?.message || String(e));
          setCurrentAnimation({
            id: selectedWord,
            name: selectedWord,
            file_path: `/animations/${selectedWord}.glb`
          });
        }
      } finally {
        if (!cancelled) setLoadingAnim(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedWord, learnPhase]);

  const resetAttemptState = useCallback(() => {
    setPracticeResult(null);
    setModelGuess(null);
    setScoreError('');
    stopSignCapture();
    recognition.clearSentence();
  }, [recognition, stopSignCapture]);

  const handleSelectWord = (word) => {
    setSelectedWord(word);
    setLearnPhase('pick');
    mediaPipe.stopDetection();
    resetAttemptState();
    setIsAnimPlaying(false);
  };

  const handleGoWatch = () => {
    if (!selectedWord || modelClasses.length === 0) return;
    resetAttemptState();
    mediaPipe.stopDetection();
    setLearnPhase('watch');
    setIsAnimPlaying(false);
  };

  const handleBackToPick = () => {
    setLearnPhase('pick');
    mediaPipe.stopDetection();
    resetAttemptState();
    setIsAnimPlaying(false);
  };

  const handleGoTest = async () => {
    if (!selectedWord) return;
    resetAttemptState();
    setLearnPhase('test');
    setIsAnimPlaying(false);
    try {
      await mediaPipe.startDetection();
      setProgress((prev) => ({ ...prev, sessions: prev.sessions + 1 }));
    } catch {
      // mediaPipe.error surfaces in UI
    }
  };

  const handleStopTest = () => {
    mediaPipe.stopDetection();
    resetAttemptState();
    setLearnPhase('watch');
  };

  const applyScoreResults = useCallback(
    (scoreRes, guessRes) => {
      setPracticeResult(scoreRes);
      setModelGuess(guessRes);
      const isCorrect = scoreRes.score >= SCORE_CORRECT_THRESHOLD;
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
    },
    [selectedWord]
  );

  const finalizeCaptureWindow = useCallback(
    async (tail60, sessionId) => {
      if (captureSessionRef.current !== sessionId) return;
      if (signCapturePhaseRef.current !== 'listening') return;

      signCapturePhaseRef.current = 'finishing';
      ingestCaptureRef.current = false;
      setSignCapturePhase('finishing');
      setIsScoring(true);
      setListenStatus(
        language === 'en' ? 'Scoring your sign…' : 'آپ کے اشارے کا اسکور…'
      );
      try {
        const validation = validateSequence(tail60);
        if (!validation.valid) {
          throw new Error(validation.error);
        }
        const hands = Math.max(1, handsDetectedRef.current);
        const [scoreRes, guessRes] = await Promise.all([
          scorePracticePSL(tail60, selectedWord, hands),
          recognizePSL(tail60, hands).catch(() => null)
        ]);
        if (captureSessionRef.current !== sessionId) return;
        applyScoreResults(scoreRes, guessRes);
        signCapturePhaseRef.current = 'done';
        setSignCapturePhase('done');
      } catch (e) {
        if (captureSessionRef.current === sessionId) {
          setScoreError(e?.message || String(e));
          signCapturePhaseRef.current = 'idle';
          setSignCapturePhase('idle');
        }
      } finally {
        if (captureSessionRef.current === sessionId) {
          setIsScoring(false);
          setListenStatus('');
        }
      }
    },
    [applyScoreResults, language, selectedWord]
  );

  /** 2s prep (no capture), then listen until stable target detection or max length */
  const handleStartSignCheck = useCallback(() => {
    if (!mediaPipe.isDetecting) {
      setScoreError(
        language === 'en' ? 'Camera is not active.' : 'کیمرہ فعال نہیں۔'
      );
      return;
    }
    if (['countdown', 'listening', 'finishing'].includes(signCapturePhase)) {
      return;
    }

    setScoreError('');
    setPracticeResult(null);
    setModelGuess(null);
    recognition.resetBuffer();
    captureRingRef.current = [];
    stableHitsRef.current = 0;
    setCaptureFrameCount(0);
    ingestCaptureRef.current = false;

    const sessionId = captureSessionRef.current + 1;
    captureSessionRef.current = sessionId;
    setCaptureSessionTick(sessionId);

    signCapturePhaseRef.current = 'countdown';
    setSignCapturePhase('countdown');
    setPrepMsLeft(CAPTURE_PREP_MS);
    setListenStatus(
      language === 'en'
        ? 'Get ready — capture starts after 2 seconds.'
        : 'تیار ہوں — 2 سیکنڈ بعد ریکارڈنگ شروع ہوگی۔'
    );
  }, [language, mediaPipe.isDetecting, recognition, signCapturePhase]);

  /** Prep countdown: no frames recorded */
  useEffect(() => {
    if (signCapturePhase !== 'countdown') return undefined;
    const sessionId = captureSessionRef.current;
    const started = Date.now();
    const id = window.setInterval(() => {
      if (captureSessionRef.current !== sessionId) {
        window.clearInterval(id);
        return;
      }
      const left = Math.max(0, CAPTURE_PREP_MS - (Date.now() - started));
      setPrepMsLeft(left);
      if (left <= 0) {
        window.clearInterval(id);
        ingestCaptureRef.current = true;
        signCapturePhaseRef.current = 'listening';
        setSignCapturePhase('listening');
        setListenStatus(
          language === 'en'
            ? 'Perform your sign — we capture until it looks complete.'
            : 'اپنا اشارہ کریں — مکمل ہونے پر خود بخود اسکور ہو گا۔'
        );
      }
    }, 80);
    return () => window.clearInterval(id);
  }, [signCapturePhase, captureSessionTick, language]);

  /** While listening: poll classifier on rolling 60-frame window until stable match or timeout */
  useEffect(() => {
    if (signCapturePhase !== 'listening') return undefined;
    const sessionId = captureSessionRef.current;

    const poll = async () => {
      if (signCapturePhaseRef.current !== 'listening') return;
      if (captureSessionRef.current !== sessionId) return;
      if (pollingBusyRef.current) return;
      const ring = captureRingRef.current;
      if (ring.length < WINDOW_FRAMES) return;
      if (handsDetectedRef.current <= 0) return;

      pollingBusyRef.current = true;
      const tail = ring.slice(-WINDOW_FRAMES);
      try {
        const r = await recognizePSL(tail, handsDetectedRef.current);
        if (captureSessionRef.current !== sessionId) return;
        if (signCapturePhaseRef.current !== 'listening') return;

        const match =
          r.label?.toLowerCase() === selectedWord.toLowerCase() && r.confidence >= STABLE_MATCH_CONF;
        if (match) {
          stableHitsRef.current += 1;
          setListenStatus(
            language === 'en'
              ? `Detected "${selectedWord}" (${(r.confidence * 100).toFixed(0)}%) — confirming…`
              : `"${selectedWord}" پہچانا گیا — تصدیق…`
          );
        } else {
          stableHitsRef.current = 0;
          setListenStatus(
            language === 'en'
              ? `Keep going… (${ring.length}/${MAX_CAPTURE_FRAMES} frames)`
              : `جاری رکھیں… (${ring.length}/${MAX_CAPTURE_FRAMES})`
          );
        }

        if (stableHitsRef.current >= STABLE_MATCH_STREAK) {
          await finalizeCaptureWindow(tail, sessionId);
          return;
        }
        if (ring.length >= MAX_CAPTURE_FRAMES) {
          await finalizeCaptureWindow(tail, sessionId);
        }
      } catch {
        stableHitsRef.current = 0;
      } finally {
        pollingBusyRef.current = false;
      }
    };

    const intervalId = window.setInterval(() => {
      void poll();
    }, LISTEN_POLL_MS);
    return () => window.clearInterval(intervalId);
  }, [signCapturePhase, captureSessionTick, selectedWord, language, finalizeCaptureWindow]);

  /** If user never fills 60 frames within wall time, cancel with a clear message */
  useEffect(() => {
    if (signCapturePhase !== 'listening') return undefined;
    const sid = captureSessionRef.current;
    const t = window.setTimeout(() => {
      if (captureSessionRef.current !== sid) return;
      if (signCapturePhaseRef.current !== 'listening') return;
      if (captureRingRef.current.length >= WINDOW_FRAMES) return;
      setScoreError(
        language === 'en'
          ? 'Not enough frames captured in time. Keep at least one hand in view and try again.'
          : 'وقت میں کافی فریمز نہیں ملے۔ ہاتھ نظر رکھیں اور دوبارہ کوشش کریں۔'
      );
      stopSignCapture();
    }, 12000);
    return () => window.clearTimeout(t);
  }, [signCapturePhase, captureSessionTick, language, stopSignCapture]);

  const handleResetProgress = useCallback(() => {
    setProgress({ attempts: 0, correct: 0, sessions: 0, practicedWords: [] });
    resetAttemptState();
  }, [resetAttemptState]);

  const averageScore = progress.attempts > 0 ? Math.round((progress.correct / progress.attempts) * 100) : 0;

  const hints = useMemo(() => {
    const baseHints = [
      language === 'en'
        ? 'Press “Check my sign”, wait 2 seconds, then perform the sign — we record until it is detected clearly or time runs out.'
        : '«میرا سائن چیک کریں» دبائیں، 2 سیکنڈ انتظار کریں، پھر اشارہ کریں — واضح پہچان یا وقت ختم ہونے تک ریکارڈ ہوتا ہے۔',
      language === 'en' ? 'Keep your hand fully visible inside the camera frame.' : 'اپنا ہاتھ کیمرے کے فریم میں مکمل طور پر رکھیں۔',
      language === 'en' ? 'Hold the sign steady for about 2 seconds within the capture window.' : 'کیپچر ونڈو میں تقریباً 2 سیکنڈ تک اشارہ مستحکم رکھیں۔',
      language === 'en' ? 'Mirror the reference hand shape and orientation.' : 'حوالہ اشارے کی ہتھیلی اور سمت کی نقالی کریں۔'
    ];
    if (!practiceResult) return baseHints;
    if (practiceResult.score >= SCORE_CORRECT_THRESHOLD) {
      return [
        language === 'en'
          ? `Strong match for "${selectedWord}". Keep practicing to stay consistent.`
          : `"${selectedWord}" کے لیے اچھی مماثلت۔ مستقل مشق جاری رکھیں۔`
      ];
    }
    const guessLine =
      modelGuess?.label && modelGuess.label.toLowerCase() !== selectedWord.toLowerCase()
        ? [
            language === 'en'
              ? `The full classifier guessed "${modelGuess.label}" — exaggerate "${selectedWord}" more clearly.`
              : `کلاسیفائر نے "${modelGuess.label}" اندازہ لگایا — "${selectedWord}" کو واضح کریں۔`
          ]
        : [];
    return [...guessLine, ...baseHints];
  }, [language, practiceResult, modelGuess, selectedWord]);

  const animPath = currentAnimation?.file_path || '';

  return (
    <div className="pt-8">
      <div className="page-wrap space-y-8 reveal-up">
        <div className="section-card p-8 sm:p-10">
          <p className="eyebrow mb-3 text-[#5f73a5]">Learning Studio</p>
          <h1 className={`title-display mb-2 text-4xl ${language === 'ur' ? 'ur-text' : ''}`}>{t('learn_title')}</h1>
          <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>{t('learn_subtitle')}</p>
          {activeApiBaseURL && (
            <p className={`mt-2 text-xs text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en' ? `Connected backend: ${activeApiBaseURL}` : `منسلک بیک اینڈ: ${activeApiBaseURL}`}
            </p>
          )}
        </div>

        <div className="stagger-grid grid gap-8 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            <div className="flex flex-wrap gap-2">
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  learnPhase === 'pick' ? 'bg-[#4f46e5] text-white' : 'bg-[#e8ecf8] text-[#5f73a5]'
                }`}
              >
                1 · {language === 'en' ? 'Choose sign' : 'اشارہ منتخب'}
              </span>
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  learnPhase === 'watch' ? 'bg-[#4f46e5] text-white' : 'bg-[#e8ecf8] text-[#5f73a5]'
                }`}
              >
                2 · {language === 'en' ? 'Watch' : 'دیکھیں'}
              </span>
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  learnPhase === 'test' ? 'bg-[#4f46e5] text-white' : 'bg-[#e8ecf8] text-[#5f73a5]'
                }`}
              >
                3 · {language === 'en' ? 'Test' : 'آزمائش'}
              </span>
            </div>

            {learnPhase === 'pick' && (
              <div className="section-card p-6 sm:p-8">
                <div className="mb-6 flex items-start gap-3">
                  <GraduationCap className="mt-1 h-8 w-8 text-[#4f46e5]" />
                  <div>
                    <h2 className={`text-lg font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en' ? 'Learn this sign' : 'یہ سائن سیکھیں'}
                    </h2>
                    <p className={`mt-1 text-sm text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en'
                        ? `Selected: "${selectedWord || '—'}". Watch the 3D demo, then test yourself against a score tailored to this word only.`
                        : `منتخب: "${selectedWord || '—'}"۔ 3D ڈیمو دیکھیں، پھر صرف اس لفظ کے لیے اسکور پر اپنی جانچ کریں۔`}
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleGoWatch}
                  disabled={!selectedWord || modelClasses.length === 0 || isLoadingModel}
                  className="btn-primary inline-flex items-center gap-2 disabled:opacity-50"
                >
                  <Play className="h-5 w-5" />
                  {language === 'en' ? 'Learn sign' : 'سائن سیکھیں'}
                </button>
              </div>
            )}

            {learnPhase === 'watch' && (
              <div className="section-card space-y-4 p-6 sm:p-8">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <button type="button" onClick={handleBackToPick} className="btn-secondary inline-flex items-center gap-2">
                    <ArrowLeft className="h-4 w-4" />
                    {language === 'en' ? 'Change word' : 'لفظ بدلیں'}
                  </button>
                  <h2 className={`text-lg font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Demonstration' : 'مظاہرہ'} — {selectedWord}
                  </h2>
                </div>

                <div className="relative aspect-video w-full overflow-hidden rounded-2xl border border-[#c8d8f2] bg-[#eef5ef]">
                  {loadingAnim ? (
                    <div className="flex h-full min-h-[280px] items-center justify-center text-[#5c6d64]">
                      {language === 'en' ? 'Loading animation…' : 'اینیمیشن لوڈ ہو رہی ہے…'}
                    </div>
                  ) : (
                    <AnimationPlayer3D
                      animationPath={animPath}
                      isPlaying={isAnimPlaying}
                      animationSpeed={animationSpeed}
                      modelScale={modelScale}
                      onAnimationComplete={() => setIsAnimPlaying(false)}
                      className="min-h-[280px]"
                    />
                  )}
                </div>
                {animError && (
                  <p className="text-sm text-amber-800">
                    {language === 'en' ? 'Animation note: ' : 'نوٹ: '}
                    {animError}
                  </p>
                )}

                <div className="grid gap-4 sm:grid-cols-2">
                  <label className="block text-sm">
                    <span className="mb-1 block font-medium text-[#1a2642]">
                      {language === 'en' ? 'Playback speed' : 'چلانے کی رفتار'}
                    </span>
                    <input
                      type="range"
                      min={0.25}
                      max={2}
                      step={0.05}
                      value={animationSpeed}
                      onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <span className="text-xs text-[#60708b]">{animationSpeed.toFixed(2)}×</span>
                  </label>
                  <label className="block text-sm">
                    <span className="mb-1 block font-medium text-[#1a2642]">
                      {language === 'en' ? 'Avatar size' : 'اوتار سائز'}
                    </span>
                    <input
                      type="range"
                      min={0.6}
                      max={2.5}
                      step={0.05}
                      value={modelScale}
                      onChange={(e) => setModelScale(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <span className="text-xs text-[#60708b]">{modelScale.toFixed(2)}×</span>
                  </label>
                </div>

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => setIsAnimPlaying(true)}
                    disabled={!animPath || loadingAnim}
                    className="btn-primary inline-flex items-center gap-2 disabled:opacity-50"
                  >
                    <Play className="h-5 w-5" />
                    {language === 'en' ? 'Play / Replay' : 'چلائیں / دوبارہ'}
                  </button>
                  <button type="button" onClick={handleGoTest} className="btn-primary inline-flex items-center gap-2 bg-[#24bf86]">
                    <ClipboardCheck className="h-5 w-5" />
                    {language === 'en' ? 'Test this sign' : 'اس سائن کی جانچ'}
                  </button>
                </div>
              </div>
            )}

            {learnPhase === 'test' && (
              <>
                <div className="overflow-hidden rounded-2xl border border-[#c8d8f2] bg-[#0b1324] shadow-lg">
                  <div className="relative flex aspect-video w-full items-center justify-center bg-black">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className={`absolute inset-0 h-full w-full -scale-x-100 transform object-cover ${
                        mediaPipe.isDetecting ? 'opacity-100' : 'opacity-0'
                      }`}
                    />
                    <canvas
                      ref={mediaPipe.canvasRef}
                      className={`absolute inset-0 h-full w-full -scale-x-100 transform object-cover ${
                        mediaPipe.isDetecting ? 'opacity-100' : 'opacity-0'
                      }`}
                    />
                    {mediaPipe.isDetecting ? (
                      <>
                        {(signCapturePhase === 'countdown' ||
                          signCapturePhase === 'listening' ||
                          signCapturePhase === 'finishing') && (
                          <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/45 px-4 text-center">
                            {signCapturePhase === 'countdown' && (
                              <>
                                <p className="text-5xl font-bold text-white drop-shadow-lg">
                                  {(prepMsLeft / 1000).toFixed(1)}s
                                </p>
                                <p className="mt-2 max-w-md text-sm font-medium text-white/95">
                                  {language === 'en'
                                    ? 'Get ready — recording starts after 2 seconds. No capture yet.'
                                    : 'تیار ہوں — 2 سیکنڈ بعد ریکارڈنگ۔ ابھی کیپچر نہیں۔'}
                                </p>
                              </>
                            )}
                            {signCapturePhase === 'listening' && (
                              <>
                                <p className="text-lg font-semibold text-white drop-shadow">
                                  {language === 'en' ? 'Recording your sign…' : 'آپ کا اشارہ ریکارڈ…'}
                                </p>
                                <p className="mt-2 max-w-md text-xs text-white/90">{listenStatus}</p>
                              </>
                            )}
                            {signCapturePhase === 'finishing' && (
                              <p className="text-lg font-semibold text-white drop-shadow">
                                {language === 'en' ? 'Scoring…' : 'اسکور…'}
                              </p>
                            )}
                          </div>
                        )}
                        <div className="absolute left-4 top-4 z-20 flex items-center gap-2 rounded-lg bg-black/55 px-3 py-2 text-sm font-medium text-white">
                          <Hand className={`h-4 w-4 ${mediaPipe.handsDetected > 0 ? 'text-green-400' : 'text-gray-300'}`} />
                          {mediaPipe.handsDetected} {language === 'en' ? 'hand(s)' : 'ہاتھ'}
                        </div>
                        <div className="absolute right-4 top-4 z-20 rounded-lg bg-black/55 px-3 py-2 text-sm font-medium text-white">
                          {mediaPipe.fps} FPS
                        </div>
                        <div className="absolute bottom-4 left-4 right-4 z-20 rounded-lg bg-black/55 p-3">
                          <div className="mb-2 flex items-center justify-between">
                            <span className="text-xs font-medium text-white">
                              {signCapturePhase === 'listening'
                                ? `${language === 'en' ? 'Session frames' : 'فریمز'}: ${captureFrameCount}/${MAX_CAPTURE_FRAMES}`
                                : signCapturePhase === 'countdown'
                                  ? language === 'en'
                                    ? 'Prep (no capture yet)'
                                    : 'تیاری (ابھی کیپچر نہیں)'
                                  : `${language === 'en' ? 'Preview buffer' : 'پیش نظارہ'}: ${recognition.bufferLength}/${WINDOW_FRAMES}`}
                            </span>
                            <span className="text-xs text-white">
                              {signCapturePhase === 'listening'
                                ? `${Math.min(100, Math.round((captureFrameCount / WINDOW_FRAMES) * 100))}%`
                                : `${recognition.bufferProgress.toFixed(0)}%`}
                            </span>
                          </div>
                          <div className="h-2 w-full rounded-full bg-[#243049]">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] transition-all duration-300"
                              style={{
                                width: `${
                                  signCapturePhase === 'listening'
                                    ? Math.min(100, (captureFrameCount / WINDOW_FRAMES) * 100)
                                    : recognition.bufferProgress
                                }%`
                              }}
                            />
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-center text-white">
                        <p className="text-lg font-medium">{language === 'en' ? 'Starting camera…' : 'کیمرہ شروع…'}</p>
                      </div>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-3 bg-[#f9fbff] px-6 py-4">
                    <button type="button" onClick={handleStopTest} className="btn-danger inline-flex flex-1 items-center justify-center gap-2 sm:flex-none">
                      <VideoOff className="h-5 w-5" />
                      {language === 'en' ? 'Back to demo' : 'ڈیمو پر واپس'}
                    </button>
                    <button
                      type="button"
                      onClick={handleStartSignCheck}
                      disabled={
                        !mediaPipe.isDetecting ||
                        ['countdown', 'listening', 'finishing'].includes(signCapturePhase)
                      }
                      className="btn-primary inline-flex flex-1 items-center justify-center gap-2 disabled:opacity-50 sm:flex-none"
                    >
                      <ClipboardCheck className="h-5 w-5" />
                      {signCapturePhase === 'countdown'
                        ? language === 'en'
                          ? 'Get ready…'
                          : 'تیار ہوں…'
                        : signCapturePhase === 'listening'
                          ? language === 'en'
                            ? 'Capturing…'
                            : 'ریکارڈ…'
                          : signCapturePhase === 'finishing'
                            ? language === 'en'
                              ? 'Scoring…'
                              : 'اسکور…'
                            : language === 'en'
                              ? 'Check my sign'
                              : 'میرا سائن چیک کریں'}
                    </button>
                    <button type="button" onClick={resetAttemptState} className="btn-secondary inline-flex items-center gap-2">
                      <RotateCcw className="h-5 w-5" />
                      {language === 'en' ? 'Reset capture' : 'کیپچر ری سیٹ'}
                    </button>
                  </div>
                </div>

                {(scoreError || practiceResult) && (
                  <div className="section-card space-y-4 p-6">
                    <h3 className={`text-lg font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {t('learn_feedback')}
                    </h3>
                    {scoreError && <p className="text-sm font-medium text-red-700">{scoreError}</p>}
                    {practiceResult && (
                      <>
                        <div className="stagger-grid grid grid-cols-2 gap-4">
                          <div className="rounded-2xl bg-[#eef3ff] p-4 text-center">
                            <p className={`mb-2 text-xs font-semibold uppercase text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                              {t('learn_reference')}
                            </p>
                            <p className="text-2xl font-bold text-[#1a2642]">{practiceResult.target_label}</p>
                            <p className="mt-1 text-xs text-[#60708b]">
                              {language === 'en' ? 'Scoring method: ' : 'طریقہ: '}
                              {practiceResult.method}
                            </p>
                          </div>
                          <div className="rounded-2xl border border-[#ffd7be] bg-[#fff2e8] p-4 text-center">
                            <p className={`mb-2 text-xs font-semibold uppercase text-[#8f6d56] ${language === 'ur' ? 'ur-text' : ''}`}>
                              {t('learn_your_attempt')}
                            </p>
                            <p className="text-2xl font-bold text-[#1a2642]">
                              {modelGuess?.label || (language === 'en' ? '—' : '—')}
                            </p>
                            <p className="mt-1 text-xs text-[#60708b]">
                              P={modelGuess != null ? (modelGuess.confidence * 100).toFixed(0) : '—'}%
                            </p>
                          </div>
                        </div>

                        <div className="rounded-2xl border border-[#d5e0f2] bg-white p-4">
                          <div className="mb-3 flex items-center justify-between">
                            <h4 className={`font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                              {t('learn_similarity')}
                            </h4>
                            <span className="text-3xl font-bold text-[#4f46e5]">{practiceResult.score}%</span>
                          </div>
                          <div className="h-3 w-full overflow-hidden rounded-full bg-[#dbe6f8]">
                            <div
                              className="h-3 rounded-full bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] transition-all duration-300"
                              style={{ width: `${Math.min(100, practiceResult.score)}%` }}
                            />
                          </div>
                          {practiceResult.cosine_similarity != null && (
                            <p className="mt-2 text-xs text-[#60708b]">
                              {language === 'en' ? 'Embedding cosine vs reference: ' : 'ریفرنس سے cosine: '}
                              {practiceResult.cosine_similarity.toFixed(4)}
                              {language === 'en' ? ' (target class softmax: ' : ' (softmax: '}
                              {(practiceResult.target_class_probability * 100).toFixed(1)}%)
                            </p>
                          )}
                          {practiceResult.cosine_similarity == null && (
                            <p className="mt-2 text-xs text-[#60708b]">
                              {language === 'en' ? 'Target-class probability: ' : 'ہدف کلاس امکان: '}
                              {(practiceResult.target_class_probability * 100).toFixed(1)}%
                            </p>
                          )}
                          <p className="mt-3 text-sm text-[#60708b]">
                            {practiceResult.score >= SCORE_CORRECT_THRESHOLD
                              ? language === 'en'
                                ? `Great work — your sign aligns well with "${selectedWord}".`
                                : `بہترین — آپ کا اشارہ "${selectedWord}" سے اچھی طرح میل کھاتا ہے۔`
                              : language === 'en'
                                ? 'Keep adjusting shape and timing; use the demo again if needed.'
                                : 'شکل اور وقت میں ترمیم کریں؛ ضرورت ہو تو ڈیمو دوبارہ دیکھیں۔'}
                          </p>
                        </div>

                        <div className="rounded-2xl border border-[#d5e0f2] bg-[#f6f9ff] p-4">
                          <h4 className={`mb-3 font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>{t('learn_hints')}</h4>
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
                      </>
                    )}
                  </div>
                )}
              </>
            )}
          </div>

          <div className="space-y-6">
            <div className="section-card p-6">
              <h3 className={`mb-4 text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                {language === 'en' ? 'Practice words' : 'سیکھنے کے الفاظ'}
              </h3>
              <div className="max-h-[420px] space-y-3 overflow-auto pr-1">
                {modelClasses.map((word) => (
                  <div
                    key={word}
                    role="button"
                    tabIndex={0}
                    onClick={() => handleSelectWord(word)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') handleSelectWord(word);
                    }}
                    className={`cursor-pointer rounded-2xl border p-4 transition-colors ${
                      selectedWord === word ? 'border-[#4f46e5] bg-[#eef3ff]' : 'border-[#d5e0f2] bg-white hover:bg-[#eef3ff]'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🖐️</span>
                      <span className="font-semibold text-[#1a2642]">{word}</span>
                    </div>
                  </div>
                ))}
                {!isLoadingModel && modelClasses.length === 0 && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                    <p className={`text-sm text-amber-800 ${language === 'ur' ? 'ur-text' : ''}`}>
                      {modelInfoError ||
                        (language === 'en' ? 'No model classes found. Start backend and retry.' : 'ماڈل کلاسز نہیں ملیں۔')}
                    </p>
                    <button type="button" onClick={fetchModelInfo} className="btn-secondary mt-2">
                      {language === 'en' ? 'Retry model sync' : 'ماڈل دوبارہ حاصل کریں'}
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="section-card p-6">
              <h3 className={`mb-4 text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                {language === 'en' ? 'Your progress' : 'آپ کی ترقی'}
              </h3>
              <div className="space-y-4">
                <div>
                  <p className={`mb-1 text-sm text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Signs practiced' : 'سیکھے گئے اشارے'}
                  </p>
                  <p className="text-3xl font-bold text-[#4f46e5]">{progress.practicedWords.length}</p>
                </div>
                <div>
                  <p className={`mb-1 text-sm text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Tests ≥72%' : 'ٹیسٹ ≥72%'}
                  </p>
                  <p className="text-3xl font-bold text-[#4f46e5]">{averageScore}%</p>
                </div>
                <div>
                  <p className={`mb-1 text-sm text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Sessions' : 'سیشنز'}
                  </p>
                  <p className="text-3xl font-bold text-[#4f46e5]">{progress.sessions}</p>
                </div>
              </div>
              <button type="button" onClick={handleResetProgress} className="btn-secondary mt-4 w-full">
                {language === 'en' ? 'Reset progress' : 'پیش رفت دوبارہ'}
              </button>
            </div>
          </div>
        </div>

        <div className="section-card p-4">
          <p className={`mb-2 text-sm font-semibold text-[#3c6fe2] ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en' ? 'How learning mode scores you' : 'لرننگ موڈ کیسے اسکور دیتا ہے'}
          </p>
          <ul className={`list-inside list-disc space-y-1 text-sm text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`}>
            <li>
              {language === 'en'
                ? 'After you tap “Check my sign”, nothing is recorded for 2 seconds so you can get into position; then capture runs until the model sees your chosen sign clearly twice in a row, or until the capture window fills.'
                : '«چیک» کے بعد 2 سیکنڈ کچھ ریکارڈ نہیں ہوتا، پھر جب تک ماڈل آپ کا منتخب لفظ دو بار مسلسل واضح نہ دیکھ لے یا ونڈو بھر جائے ریکارڈنگ جاری رہتی ہے۔'}
            </li>
            <li>
              {language === 'en'
                ? 'Watch the same GLB animation used in Text-to-PSL, then test with the camera.'
                : 'Text-to-PSL جیسی GLB دیکھیں، پھر کیمرے سے جانچیں۔'}
            </li>
            <li>
              {language === 'en'
                ? 'Your score compares your motion to a stored reference for only the word you chose (embedding cosine), not picking among all 32 signs.'
                : 'آپ کا اسکور صرف منتخب لفظ کے محفوظ حوالے سے موازنہ ہے، 32 میں سے انتخاب نہیں۔'}
            </li>
            <li>
              {language === 'en'
                ? 'If prototypes are unavailable, the server falls back to the target class probability only.'
                : 'اگر پروٹو ٹائپ نہ ہوں تو سرور صرف ہدف کلاس کا امکان استعمال کرے گا۔'}
            </li>
          </ul>
          {(mediaPipe.error || recognition.error) && (
            <p className="mt-3 text-sm font-semibold text-red-700">
              {language === 'en' ? 'Error: ' : 'خرابی: '}
              {mediaPipe.error || recognition.error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
