import { Sparkles, User, Camera } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface HomeProps {
  onNavigate: (screen: 'home' | 'analysis' | 'recommendations' | 'profile' | 'history') => void;
}

export function Home({ onNavigate }: HomeProps) {
  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-[#44475A] to-[#6272A4]">
      {/* Header */}
      <div className="flex justify-between items-center px-6 pt-14 pb-6">
        <div>
          <h1 className="text-white">Welcome back, Maya</h1>
          <p className="text-white/70 text-sm mt-1">Let's find your perfect glow ✨</p>
        </div>
        <button 
          onClick={() => onNavigate('profile')}
          className="w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center"
        >
          <User className="w-6 h-6 text-white" />
        </button>
      </div>

      {/* Main Content */}
      <div className="flex-1 px-6 pb-6 overflow-y-auto">
        {/* Hero Card */}
        <div className="bg-white/95 backdrop-blur-sm rounded-[32px] p-6 mb-6 shadow-xl">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center">
              <Camera className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h2>Discover Your Skin Tone</h2>
              <p className="text-sm text-gray-600">Get personalized recommendations</p>
            </div>
          </div>
          <button 
            onClick={() => onNavigate('analysis')}
            className="w-full bg-gradient-to-r from-[#BD93F9] to-[#FF79C6] text-white py-4 rounded-[20px] transition-transform active:scale-95"
          >
            Start Analysis
          </button>
        </div>

        {/* Featured Section */}
        <div className="mb-6">
          <h3 className="text-white mb-4">Celebrate Your Unique Beauty</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="relative h-44 rounded-[24px] overflow-hidden">
              <ImageWithFallback 
                src="https://images.unsplash.com/photo-1572568895161-ba8446444d5a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkaXZlcnNlJTIwd29tZW4lMjBiZWF1dHl8ZW58MXx8fHwxNzYyNjIxMzA1fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                alt="Diverse beauty"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
              <p className="absolute bottom-3 left-3 right-3 text-white text-sm">All Skin Tones Welcome</p>
            </div>
            <div className="relative h-44 rounded-[24px] overflow-hidden">
              <ImageWithFallback 
                src="https://images.unsplash.com/photo-1645538297959-24ed010bd775?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b21hbiUyMGdsb3dpbmclMjBza2lufGVufDF8fHx8MTc2MjUzMjM5Nnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                alt="Glowing skin"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
              <p className="absolute bottom-3 left-3 right-3 text-white text-sm">Radiant Results</p>
            </div>
          </div>
        </div>

        {/* Quick Tips */}
        <div className="bg-white/10 backdrop-blur-sm rounded-[24px] p-5 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-5 h-5 text-[#FF79C6]" />
            <h3 className="text-white">Today's Tip</h3>
          </div>
          <p className="text-white/90 text-sm leading-relaxed">
            Hydration is key! Drink plenty of water and use a moisturizer suited to your skin type for that healthy glow. ✨
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white/10 backdrop-blur-sm rounded-[20px] p-4">
            <p className="text-[#BD93F9] mb-1">Your Routine</p>
            <p className="text-white">7 Products</p>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-[20px] p-4">
            <p className="text-[#FF79C6] mb-1">Skin Health</p>
            <p className="text-white">Great! 🌟</p>
          </div>
        </div>
      </div>

      {/* Bottom Navigation */}
      <div className="bg-white/95 backdrop-blur-sm px-8 py-4 flex justify-around items-center border-t border-gray-200/50">
        <button className="flex flex-col items-center gap-1">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center">
            <div className="w-6 h-6 rounded-full bg-white" />
          </div>
          <span className="text-xs text-gray-900">Home</span>
        </button>
        <button 
          onClick={() => onNavigate('analysis')}
          className="flex flex-col items-center gap-1"
        >
          <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
            <Camera className="w-5 h-5 text-gray-600" />
          </div>
          <span className="text-xs text-gray-600">Analyze</span>
        </button>
        <button 
          onClick={() => onNavigate('recommendations')}
          className="flex flex-col items-center gap-1"
        >
          <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-gray-600" />
          </div>
          <span className="text-xs text-gray-600">For You</span>
        </button>
        <button 
          onClick={() => onNavigate('profile')}
          className="flex flex-col items-center gap-1"
        >
          <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
            <User className="w-5 h-5 text-gray-600" />
          </div>
          <span className="text-xs text-gray-600">Profile</span>
        </button>
      </div>
    </div>
  );
}