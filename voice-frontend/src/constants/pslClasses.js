export const PSL_CLASS_FALLBACK = [
  'alert',
  'book',
  'careful',
  'cheap',
  'crazy',
  'dangerous',
  'decent',
  'dumb',
  'excited',
  'extreme',
  'fantastic',
  'far',
  'fearful',
  'foreign',
  'funny',
  'good',
  'healthy',
  'heavy',
  'important',
  'intelligent',
  'interesting',
  'late',
  'less',
  'new',
  'no',
  'noisy',
  'peaceful',
  'quick',
  'ready',
  'secure',
  'smart',
  'yes'
];

export const normalizePSLClasses = (info) => {
  if (Array.isArray(info?.classes) && info.classes.length > 0) {
    return info.classes;
  }

  if (info?.loaded && Number(info?.num_classes) === PSL_CLASS_FALLBACK.length) {
    return PSL_CLASS_FALLBACK;
  }

  return [];
};
