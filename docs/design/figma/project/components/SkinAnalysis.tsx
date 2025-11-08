import { useState } from 'react';
import { ArrowLeft, Camera, Sparkles, Check, Zap, RefreshCw, HelpCircle } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface SkinAnalysisProps {
  onNavigate: (screen: 'home' | 'analysis' | 'recommendations' | 'profile' | 'history') => void;
  onAnalysisComplete: (skinTone: string) => void;
}

export function SkinAnalysis({ onNavigate, onAnalysisComplete }: SkinAnalysisProps) {
  const [analyzing, setAnalyzing] = useState(false);
  const [step, setStep] = useState<'instructions' | 'camera' | 'analyzing' | 'complete'>('instructions');
  const [lightingQuality, setLightingQuality] = useState(75); // 0-100
  const [flashOn, setFlashOn] = useState(false);
  const [detectedMonkScale, setDetectedMonkScale] = useState(6);
  const [confidence, setConfidence] = useState(92);

  const monkScaleTones = [
    { scale: 1, color: '#F6EDE4', name: 'Lightest' },
    { scale: 2, color: '#F3E7DB', name: 'Very Light' },
    { scale: 3, color: '#F7DED0', name: 'Light' },
    { scale: 4, color: '#ECCABA', name: 'Light-Medium' },
    { scale: 5, color: '#D5BA9D', name: 'Medium' },
    { scale: 6, color: '#C09873', name: 'Medium-Tan' },
    { scale: 7, color: '#A57E5C', name: 'Tan' },
    { scale: 8, color: '#7A4820', name: 'Deep' },
    { scale: 9, color: '#5D3F1A', name: 'Very Deep' },
    { scale: 10, color: '#3A2416', name: 'Deepest' },
  ];

  const skinTones = [
    { name: 'Fair', color: '#F5D7C3', description: 'Light with cool undertones' },
    { name: 'Light', color: '#E8B896', description: 'Light with warm undertones' },
    { name: 'Medium', color: '#C68642', description: 'Medium with golden undertones' },
    { name: 'Olive', color: '#A67C52', description: 'Medium with olive undertones' },
    { name: 'Tan', color: '#8D5524', description: 'Deep with warm undertones' },
    { name: 'Deep', color: '#5C3317', description: 'Rich with cool undertones' },
  ];

  const handleStartAnalysis = () => {
    setStep('camera');
    setTimeout(() => {
      setStep('analyzing');
      setAnalyzing(true);
      setTimeout(() => {
        setAnalyzing(false);
        setStep('complete');
      }, 2500);
    }, 1500);
  };

  const handleSelectTone = (tone: string) => {
    onAnalysisComplete(tone);
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-[#44475A] to-[#6272A4]">
      {/* Header - Only show for non-camera steps */}
      {step !== 'camera' && (
        <div className="flex items-center gap-4 px-6 pt-14 pb-6">
          <button 
            onClick={() => onNavigate('home')}
            className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center"
          >
            <ArrowLeft className="w-5 h-5 text-white" />
          </button>
          <h1 className="text-white">Skin Tone Analysis</h1>
        </div>
      )}

      {/* Content */}
      {step === 'camera' ? (
        /* Camera Full Screen View */
        <div className="h-full flex flex-col bg-gradient-to-br from-[#44475A] to-[#6272A4]">
          {/* Back Button Overlay */}
          <div className="absolute top-14 left-6 z-10">
            <button 
              onClick={() => onNavigate('home')}
              className="w-10 h-10 rounded-full bg-black/30 backdrop-blur-sm flex items-center justify-center"
            >
              <ArrowLeft className="w-5 h-5 text-white" />
            </button>
          </div>

          {/* Top Instruction Text */}
          <div className="px-6 pt-20 pb-8 text-center">
            <p className="text-white">Find good lighting and center your face</p>
          </div>

          {/* Camera Preview with Face Guide */}
          <div className="flex-1 flex items-center justify-center px-6">
            <div className="relative w-full max-w-[280px]">
              {/* Face Guide - Large Rounded Rectangle */}
              <div className="relative aspect-[3/4] w-full">
                <div className="absolute inset-0 rounded-[48px] border-[3px] border-dashed border-white/60" />
                
                {/* Corner Guides */}
                <div className="absolute top-0 left-0 w-12 h-12">
                  <div className="absolute top-0 left-0 w-12 h-1 bg-gradient-to-r from-[#BD93F9] to-transparent rounded-full" />
                  <div className="absolute top-0 left-0 w-1 h-12 bg-gradient-to-b from-[#BD93F9] to-transparent rounded-full" />
                </div>
                <div className="absolute top-0 right-0 w-12 h-12">
                  <div className="absolute top-0 right-0 w-12 h-1 bg-gradient-to-l from-[#FF79C6] to-transparent rounded-full" />
                  <div className="absolute top-0 right-0 w-1 h-12 bg-gradient-to-b from-[#FF79C6] to-transparent rounded-full" />
                </div>
                <div className="absolute bottom-0 left-0 w-12 h-12">
                  <div className="absolute bottom-0 left-0 w-12 h-1 bg-gradient-to-r from-[#BD93F9] to-transparent rounded-full" />
                  <div className="absolute bottom-0 left-0 w-1 h-12 bg-gradient-to-t from-[#BD93F9] to-transparent rounded-full" />
                </div>
                <div className="absolute bottom-0 right-0 w-12 h-12">
                  <div className="absolute bottom-0 right-0 w-12 h-1 bg-gradient-to-l from-[#FF79C6] to-transparent rounded-full" />
                  <div className="absolute bottom-0 right-0 w-1 h-12 bg-gradient-to-t from-[#FF79C6] to-transparent rounded-full" />
                </div>
              </div>

              {/* Center Helper Text */}
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-white/50 text-sm">Align your face here</p>
              </div>
            </div>
          </div>

          {/* Lighting Quality Indicator */}
          <div className="px-6 pb-6">
            <div className="bg-[#44475A]/80 backdrop-blur-sm rounded-[20px] p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white text-sm">Lighting Quality:</span>
                <span className="text-white">
                  {lightingQuality >= 70 ? 'Good' : lightingQuality >= 40 ? 'Fair' : 'Poor'}
                </span>
              </div>
              {/* Progress Bar */}
              <div className="h-2 bg-white/20 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-[#BD93F9] to-[#FF79C6] transition-all duration-300 rounded-full"
                  style={{ width: `${lightingQuality}%` }}
                />
              </div>
            </div>
          </div>

          {/* Control Buttons Row */}
          <div className="px-6 pb-12 flex items-center justify-center gap-6">
            {/* Flash Button */}
            <button
              onClick={() => setFlashOn(!flashOn)}
              className={`w-14 h-14 rounded-full backdrop-blur-sm flex items-center justify-center transition-colors ${
                flashOn ? 'bg-white/30' : 'bg-white/10'
              }`}
            >
              <Zap className={`w-6 h-6 ${flashOn ? 'text-yellow-300 fill-current' : 'text-white'}`} />
            </button>

            {/* Capture Button - Large with Glow */}
            <button
              onClick={() => {
                setStep('analyzing');
                setAnalyzing(true);
                setTimeout(() => {
                  setAnalyzing(false);
                  setStep('complete');
                }, 2500);
              }}
              className="relative w-20 h-20 flex items-center justify-center transition-transform active:scale-95"
            >
              {/* Purple Glow Effect */}
              <div className="absolute inset-0 rounded-full bg-[#BD93F9] opacity-40 blur-xl" />
              
              {/* Outer Ring */}
              <div className="absolute inset-0 rounded-full border-4 border-white/30" />
              
              {/* Inner Button */}
              <div className="relative w-16 h-16 rounded-full bg-gradient-to-br from-[#FF79C6] to-[#BD93F9] shadow-2xl" />
            </button>

            {/* Flip Camera Button */}
            <button
              className="w-14 h-14 rounded-full bg-white/10 backdrop-blur-sm flex items-center justify-center"
            >
              <RefreshCw className="w-6 h-6 text-white" />
            </button>
          </div>
        </div>
      ) : (
        <div className="flex-1 px-6 pb-6 overflow-y-auto">
          {step === 'instructions' && (
            <div className="space-y-6">
              <div className="bg-white/95 backdrop-blur-sm rounded-[32px] p-6">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center mx-auto mb-4">
                  <Camera className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-center mb-2">Let's Find Your Perfect Match</h2>
                <p className="text-center text-gray-600 text-sm">
                  Our AI will analyze your skin tone to provide personalized product recommendations
                </p>
              </div>

              <div className="bg-white/95 backdrop-blur-sm rounded-[24px] p-6 space-y-4">
                <h3 className="text-gray-900">For Best Results:</h3>
                <div className="space-y-3">
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-[#BD93F9]/20 flex items-center justify-center flex-shrink-0">
                      <span className="text-[#BD93F9]">1</span>
                    </div>
                    <div>
                      <p className="text-gray-900">Natural Lighting</p>
                      <p className="text-sm text-gray-600">Stand near a window or outdoors</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-[#FF79C6]/20 flex items-center justify-center flex-shrink-0">
                      <span className="text-[#FF79C6]">2</span>
                    </div>
                    <div>
                      <p className="text-gray-900">Clean Face</p>
                      <p className="text-sm text-gray-600">Remove makeup for accurate results</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-[#BD93F9]/20 flex items-center justify-center flex-shrink-0">
                      <span className="text-[#BD93F9]">3</span>
                    </div>
                    <div>
                      <p className="text-gray-900">Steady Camera</p>
                      <p className="text-sm text-gray-600">Hold your phone still for 3 seconds</p>
                    </div>
                  </div>
                </div>
              </div>

              <button 
                onClick={handleStartAnalysis}
                className="w-full bg-gradient-to-r from-[#BD93F9] to-[#FF79C6] text-white py-4 rounded-[20px] transition-transform active:scale-95"
              >
                I'm Ready!
              </button>
            </div>
          )}

          {step === 'analyzing' && (
            <div className="space-y-6">
              <div className="bg-white/95 backdrop-blur-sm rounded-[32px] p-8 flex flex-col items-center justify-center min-h-[400px]">
                <div className="relative w-32 h-32 mb-6">
                  <div className="absolute inset-0 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] animate-pulse" />
                  <div className="absolute inset-2 rounded-full bg-white flex items-center justify-center">
                    <Sparkles className="w-12 h-12 text-[#BD93F9] animate-spin" />
                  </div>
                </div>
                <h2 className="text-center mb-2">Analyzing Your Skin Tone</h2>
                <p className="text-center text-gray-600 text-sm">
                  This will only take a moment...
                </p>
              </div>
            </div>
          )}

          {step === 'complete' && (
            <div className="space-y-6 pb-8">
              {/* User Photo */}
              <div className="flex justify-center">
                <div className="relative w-48 h-48 rounded-[32px] overflow-hidden shadow-2xl">
                  <ImageWithFallback 
                    src="https://images.unsplash.com/photo-1737978697863-5d65495b28ef?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b21hbiUyMG1lZGl1bSUyMHNraW4lMjB0b25lJTIwcG9ydHJhaXR8ZW58MXx8fHwxNzYyNjI3NjQ0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                    alt="Your photo"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>

              {/* Heading */}
              <h2 className="text-center text-white">Your Skin Tone</h2>

              {/* Monk Skin Tone Scale Card */}
              <div className="bg-white/95 backdrop-blur-sm rounded-[32px] p-6">
                <div className="mb-6">
                  <p className="text-center text-gray-600 text-sm mb-4">Monk Skin Tone Scale</p>
                  
                  {/* Horizontal Color Bar */}
                  <div className="flex gap-1 mb-4">
                    {monkScaleTones.map((tone) => (
                      <div
                        key={tone.scale}
                        className={`relative flex-1 h-12 rounded-lg transition-all ${
                          tone.scale === detectedMonkScale ? 'ring-4 ring-[#BD93F9] shadow-lg' : ''
                        }`}
                        style={{ 
                          backgroundColor: tone.color,
                          boxShadow: tone.scale === detectedMonkScale ? '0 0 30px rgba(189, 147, 249, 0.6)' : 'none'
                        }}
                      >
                        {tone.scale === detectedMonkScale && (
                          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
                            <div className="w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-[#BD93F9]" />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  
                  {/* Scale Numbers */}
                  <div className="flex justify-between px-1 mb-6">
                    {monkScaleTones.map((tone) => (
                      <span
                        key={tone.scale}
                        className={`text-xs ${
                          tone.scale === detectedMonkScale ? 'text-[#BD93F9]' : 'text-gray-400'
                        }`}
                      >
                        {tone.scale}
                      </span>
                    ))}
                  </div>

                  {/* Monk Scale Label */}
                  <div className="text-center mb-6">
                    <p className="text-gray-900 mb-1">Monk Scale {detectedMonkScale}</p>
                    <p className="text-sm text-gray-600">
                      {monkScaleTones.find(t => t.scale === detectedMonkScale)?.name}
                    </p>
                  </div>

                  {/* Confidence Score with Circular Progress */}
                  <div className="flex items-center justify-center gap-4">
                    <div className="relative w-24 h-24">
                      {/* Background Circle */}
                      <svg className="w-24 h-24 transform -rotate-90">
                        <circle
                          cx="48"
                          cy="48"
                          r="40"
                          stroke="#f0f0f0"
                          strokeWidth="6"
                          fill="none"
                        />
                        {/* Progress Circle */}
                        <circle
                          cx="48"
                          cy="48"
                          r="40"
                          stroke="url(#gradient)"
                          strokeWidth="6"
                          fill="none"
                          strokeLinecap="round"
                          strokeDasharray={`${2 * Math.PI * 40}`}
                          strokeDashoffset={`${2 * Math.PI * 40 * (1 - confidence / 100)}`}
                          className="transition-all duration-1000"
                        />
                        <defs>
                          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#BD93F9" />
                            <stop offset="100%" stopColor="#FF79C6" />
                          </linearGradient>
                        </defs>
                      </svg>
                      {/* Center Text */}
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-gray-900">{confidence}%</span>
                        <span className="text-xs text-gray-600">Match</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-gray-900">High Confidence</p>
                      <p className="text-sm text-gray-600">Your results are ready!</p>
                    </div>
                  </div>
                </div>

                {/* Congratulatory Message */}
                <div className="bg-gradient-to-r from-[#BD93F9]/10 to-[#FF79C6]/10 rounded-[20px] p-4 mb-6">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-5 h-5 text-[#BD93F9]" />
                    <p className="text-gray-900">Beautiful!</p>
                  </div>
                  <p className="text-sm text-gray-600">
                    We've identified your unique skin tone and have personalized recommendations ready just for you.
                  </p>
                </div>

                {/* Buttons */}
                <div className="space-y-3">
                  {/* View Recommendations Button */}
                  <button
                    onClick={() => handleSelectTone('Medium-Tan')}
                    className="w-full bg-gradient-to-r from-[#FF79C6] to-[#BD93F9] text-white py-4 rounded-[20px] transition-transform active:scale-95"
                  >
                    View Recommendations
                  </button>

                  {/* Retake Photo Button */}
                  <button
                    onClick={() => setStep('camera')}
                    className="w-full border-2 border-white text-gray-900 py-4 rounded-[20px] transition-colors hover:bg-gray-50"
                  >
                    Retake Photo
                  </button>

                  {/* How does this work link */}
                  <button className="w-full flex items-center justify-center gap-2 py-2 text-gray-600 text-sm">
                    <HelpCircle className="w-4 h-4" />
                    How does this work?
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}