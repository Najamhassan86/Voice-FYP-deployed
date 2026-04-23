import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { usePSLRecognition } from '../hooks/usePSLRecognition';
import { getPSLModelInfo } from '../api/pslApi';
import { Volume2, Trash2, RotateCcw, Video, VideoOff, Hand } from 'lucide-react';

export const PSLtoText = () => {
  const { t, language } = useLanguage();
  const videoRef = useRef(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [modelClasses, setModelClasses] = useState([]);

  // MediaPipe hand detection
  const mediaPipe = useMediaPipe(videoRef, {
    onFeatures: (features) => {
      // Add features to PSL recognition buffer
      recognition.addFrame(features);
    },
    drawLandmarks: true
  });

  // PSL recognition with auto-recognition
  const recognition = usePSLRecognition({
    sequenceLength: 60,
    confidenceThreshold: 0.6,
    cooldownMs: 1500,
    autoRecognize: true,
    allowedLabels: modelClasses
  });

  useEffect(() => {
    const loadModelInfo = async () => {
      try {
        const info = await getPSLModelInfo();
        if (info?.loaded && Array.isArray(info.classes)) {
          setModelClasses(info.classes);
        }
      } catch (err) {
        console.warn('Could not fetch model info:', err.message);
      }
    };

    loadModelInfo();
  }, []);

  // Camera control
  const handleStartCamera = useCallback(async () => {
    try {
      await mediaPipe.startDetection();
    } catch (err) {
      console.error('Failed to start camera:', err);
    }
  }, [mediaPipe]);

  const handleStopCamera = useCallback(() => {
    mediaPipe.stopDetection();
    recognition.resetBuffer();
  }, [mediaPipe, recognition]);

  // Text-to-Speech
  const handleSpeak = useCallback(() => {
    const text = recognition.getSentenceText();
    if (!text) return;

    setIsSpeaking(true);
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.lang = language === 'ur' ? 'ur-PK' : 'en-US';

    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, [recognition, language]);

  // Clear sentence
  const handleClearSentence = useCallback(() => {
    window.speechSynthesis.cancel();
    recognition.clearSentence();
    setIsSpeaking(false);
  }, [recognition]);

  // Reset buffer
  const handleResetBuffer = useCallback(() => {
    recognition.resetBuffer();
  }, [recognition]);

  return (
    <div className="pt-8">
      <div className="page-wrap space-y-8 reveal-up">
        <div className="section-card p-8 sm:p-10">
          <p className="eyebrow mb-3 text-[#5f73a5]">Live Capture</p>
          <h1 className={`title-display mb-2 text-4xl ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en' ? 'PSL to Text & Speech' : 'PSL سے متن اور آواز'}
          </h1>
          <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en'
              ? 'Capture hand signs and convert them into real-time text with speech output.'
              : 'ہاتھ کے اشاروں کو حقیقی وقت میں متن اور آواز میں تبدیل کریں۔'}
          </p>
          {modelClasses.length > 0 && (
            <p className={`mt-3 text-xs text-[#4f5f88] ${language === 'ur' ? 'ur-text' : ''}`}>
              {language === 'en'
                ? `Active model classes (${modelClasses.length}): ${modelClasses.join(', ')}`
                : `فعال ماڈل کلاسز (${modelClasses.length}): ${modelClasses.join(', ')}`}
            </p>
          )}
        </div>

        <div className="stagger-grid grid gap-8 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            <div className="relative aspect-video overflow-hidden rounded-2xl border border-[#c8d8f2] bg-[#0b1324]">
              <video
                ref={videoRef}
                className="absolute inset-0 w-full h-full object-cover transform -scale-x-100"
                autoPlay
                playsInline
                muted
              />

              <canvas
                ref={mediaPipe.canvasRef}
                className="absolute inset-0 w-full h-full transform -scale-x-100"
              />

              {!mediaPipe.isDetecting && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                  <div className="text-center text-white">
                    <Video className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-semibold">
                      {language === 'en' ? 'Camera Inactive' : 'کیمرہ غیر فعال'}
                    </p>
                  </div>
                </div>
              )}

              {mediaPipe.isDetecting && (
                <div className="absolute left-4 top-4 flex items-center gap-2 rounded-lg bg-black/55 px-3 py-2">
                  <Hand className={`w-5 h-5 ${mediaPipe.handsDetected > 0 ? 'text-green-400' : 'text-gray-400'}`} />
                  <span className="text-white text-sm font-medium">
                    {mediaPipe.handsDetected} {language === 'en' ? 'Hand(s)' : 'ہاتھ'}
                  </span>
                </div>
              )}

              {mediaPipe.isDetecting && (
                <div className="absolute right-4 top-4 rounded-lg bg-black/55 px-3 py-2">
                  <span className="text-white text-sm font-medium">
                    {mediaPipe.fps} FPS
                  </span>
                </div>
              )}

              {mediaPipe.isDetecting && (
                <div className="absolute bottom-4 left-4 right-4">
                  <div className="rounded-lg bg-black/55 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white text-xs font-medium">
                        {language === 'en' ? 'Buffer' : 'بفر'}: {recognition.bufferLength}/60
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
                </div>
              )}

              {recognition.isRecognizing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                  <div className="flex items-center gap-2 rounded-full bg-[#4f46e5]/90 px-6 py-3 font-semibold text-white">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    {language === 'en' ? 'Recognizing...' : 'شناخت کر رہے ہیں...'}
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-3">
              {!mediaPipe.isDetecting ? (
                <button
                  onClick={handleStartCamera}
                  disabled={mediaPipe.isInitialized && mediaPipe.isDetecting}
                  className="btn-primary flex-1 disabled:opacity-50"
                >
                  <Video className="w-5 h-5" />
                  {language === 'en' ? 'Start Camera' : 'کیمرہ شروع کریں'}
                </button>
              ) : (
                <button
                  onClick={handleStopCamera}
                  className="btn-danger flex-1"
                >
                  <VideoOff className="w-5 h-5" />
                  {language === 'en' ? 'Stop Camera' : 'کیمرہ بند کریں'}
                </button>
              )}

              <button
                onClick={handleResetBuffer}
                disabled={!mediaPipe.isDetecting}
                className="btn-secondary disabled:cursor-not-allowed disabled:opacity-50"
              >
                <RotateCcw className="w-5 h-5" />
                {language === 'en' ? 'Reset' : 'دوبارہ ترتیب'}
              </button>
            </div>

            {!mediaPipe.isDetecting && (
              <div className="section-card p-4">
                <h3 className={`mb-2 font-semibold text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                  💡 {language === 'en' ? 'How to Use' : 'استعمال کیسے کریں'}
                </h3>
                <ul className={`list-inside list-disc space-y-1 text-sm text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`}>
                  <li>{language === 'en' ? 'Click "Start Camera" to begin' : 'شروع کرنے کے لیے "کیمرہ شروع کریں" پر کلک کریں'}</li>
                  <li>{language === 'en' ? 'Perform PSL signs clearly in front of the camera' : 'کیمرے کے سامنے واضح طور پر PSL اشارے کریں'}</li>
                  <li>{language === 'en' ? 'Hold each sign for 2 seconds' : 'ہر اشارہ 2 سیکنڈ تک رکھیں'}</li>
                  <li>{language === 'en' ? 'Recognized words will appear in the prediction panel' : 'پہچانے گئے الفاظ پیشین گوئی پینل میں ظاہر ہوں گے'}</li>
                </ul>
              </div>
            )}

            {(mediaPipe.error || recognition.error) && (
              <div className="rounded-2xl border border-red-200 bg-red-50 p-4">
                <p className="text-sm text-red-800 font-semibold">
                  ⚠️ Error: {mediaPipe.error || recognition.error}
                </p>
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="section-card space-y-4 p-6">
              <h3 className={`text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                {language === 'en' ? 'Current Sign' : 'موجودہ اشارہ'}
              </h3>

              {recognition.currentPrediction ? (
                <>
                  <div className="text-center py-6">
                    <p className={`mb-2 text-4xl font-bold text-[#4f46e5] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {recognition.currentPrediction.label}
                    </p>
                    <p className={`text-sm text-[#5c6d64] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en' ? 'Confidence' : 'اعتماد'}:{' '}
                      {(recognition.currentPrediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  <div>
                    <h4 className={`mb-2 text-xs font-semibold uppercase text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en' ? 'Top Predictions' : 'اوپر کی پیشین گوئیاں'}
                    </h4>
                    <div className="space-y-2">
                      {recognition.currentPrediction.top_predictions?.slice(0, 3).map((pred, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <div className="h-2 flex-1 rounded-full bg-[#dbe6f8]">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-[#4f46e5] via-[#2ea6ff] to-[#24bf86] transition-all"
                              style={{ width: `${pred.confidence * 100}%` }}
                            />
                          </div>
                          <span className="w-20 text-right text-xs font-medium text-[#2d3c54]">
                            {pred.label}
                          </span>
                          <span className="w-12 text-right text-xs text-[#60708b]">
                            {(pred.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-8 text-center text-[#819087]">
                  <p className={`text-sm ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'Waiting for recognition...' : 'شناخت کا انتظار...'}
                  </p>
                </div>
              )}
            </div>

            <div className="section-card space-y-4 p-6">
              <h3 className={`text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                {language === 'en' ? 'Sentence' : 'جملہ'}
              </h3>

              <div className="min-h-[100px] rounded-2xl border border-[#d5e0f2] bg-white p-4">
                {recognition.sentence.length > 0 ? (
                  <p className={`text-lg leading-relaxed text-[#1a2642] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {recognition.sentence.join(' ')}
                  </p>
                ) : (
                  <p className={`text-sm italic text-[#819087] ${language === 'ur' ? 'ur-text' : ''}`}>
                    {language === 'en' ? 'No words yet...' : 'ابھی تک کوئی لفظ نہیں...'}
                  </p>
                )}
              </div>

              <div className={`flex items-center justify-between text-xs text-[#60708b] ${language === 'ur' ? 'ur-text' : ''}`}>
                <span>
                  {recognition.sentence.length} {language === 'en' ? 'word(s)' : 'لفظ'}
                </span>
                <span>
                  {recognition.getSentenceText().length} {language === 'en' ? 'character(s)' : 'حروف'}
                </span>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleSpeak}
                  disabled={recognition.sentence.length === 0 || isSpeaking}
                  className="btn-primary flex-1 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Volume2 className="w-4 h-4" />
                  {isSpeaking
                    ? (language === 'en' ? 'Speaking...' : 'بول رہے ہیں...')
                    : (language === 'en' ? 'Speak' : 'بولیں')}
                </button>

                <button
                  onClick={recognition.undoLastWord}
                  disabled={recognition.sentence.length === 0}
                  className="btn-secondary px-4 disabled:cursor-not-allowed disabled:opacity-50"
                  title={language === 'en' ? 'Undo last word' : 'آخری لفظ واپس کریں'}
                >
                  <RotateCcw className="w-4 h-4" />
                </button>

                <button
                  onClick={handleClearSentence}
                  disabled={recognition.sentence.length === 0}
                  className="btn-danger px-4 disabled:cursor-not-allowed disabled:opacity-50"
                  title={language === 'en' ? 'Clear sentence' : 'جملہ صاف کریں'}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            {mediaPipe.isDetecting && (
              <div className="section-card space-y-4 p-6">
                <h3 className={`text-sm font-semibold uppercase tracking-[0.14em] text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                  {language === 'en' ? 'Live Statistics' : 'براہ راست اعداد و شمار'}
                </h3>
                <div className="space-y-3">
                  <div>
                    <p className={`mb-1 text-xs font-semibold uppercase text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en' ? 'Buffer Status' : 'بفر کی حالت'}
                    </p>
                    <p className="text-2xl font-bold text-[#4f46e5]">
                      {recognition.bufferLength}/60
                    </p>
                  </div>
                  <div>
                    <p className={`mb-1 text-xs font-semibold uppercase text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en' ? 'Words Recognized' : 'پہچانے گئے الفاظ'}
                    </p>
                    <p className="text-lg font-semibold text-[#1a2642]">
                      {recognition.sentence.length}
                    </p>
                  </div>
                  <div>
                    <p className={`mb-1 text-xs font-semibold uppercase text-[#5f73a5] ${language === 'ur' ? 'ur-text' : ''}`}>
                      {language === 'en' ? 'Hands Detected' : 'ہاتھ پائے گئے'}
                    </p>
                    <p className="text-lg font-semibold text-[#1a2642]">
                      {mediaPipe.handsDetected}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="section-card p-5">
          <p className={`mb-2 text-sm font-semibold text-[#3c6fe2] ${language === 'ur' ? 'ur-text' : ''}`}>
            ✅ {language === 'en' ? '32 PSL Signs Available for Recognition' : '32 PSL اشارے شناخت کے لیے دستیاب ہیں'}
          </p>
          <p className={`text-sm text-[#2d3c54] ${language === 'ur' ? 'ur-text' : ''}`}>
            {language === 'en'
              ? 'Available signs: alert, book, careful, cheap, crazy, dangerous, decent, dumb, excited, extreme, fantastic, far, fearful, foreign, funny, good, healthy, heavy, important, intelligent, interesting, late, less, new, no, noisy, peaceful, quick, ready, secure, smart, yes'
              : 'دستیاب اشارے: الرٹ، کتاب، محتاط، سستا، پاگل، خطرناک، مہذب، احمق، خوش، انتہائی، بہترین، دور، خوفزدہ، غیر ملکی، مضحکہ خیز، اچھا، صحت مند، بھاری، اہم، عقلمند، دلچسپ، دیر، کم، نیا، نہیں، شور، پرامن، تیز، تیار، محفوظ، ذہین، ہاں'}
          </p>
        </div>
      </div>
    </div>
  );
};
