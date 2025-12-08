import { useState } from 'react';
import { Home } from './components/Home';
import { SkinAnalysis } from './components/SkinAnalysis';
import { Recommendations } from './components/Recommendations';
import { Profile } from './components/Profile';
import { History } from './components/History';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<'home' | 'analysis' | 'recommendations' | 'profile' | 'history'>('history');
  const [skinTone, setSkinTone] = useState<string | null>(null);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="relative w-[390px] h-[844px] bg-white overflow-hidden shadow-2xl">
        {currentScreen === 'home' && (
          <Home 
            onNavigate={setCurrentScreen}
          />
        )}
        {currentScreen === 'analysis' && (
          <SkinAnalysis 
            onNavigate={setCurrentScreen}
            onAnalysisComplete={(tone) => {
              setSkinTone(tone);
              setCurrentScreen('recommendations');
            }}
          />
        )}
        {currentScreen === 'recommendations' && (
          <Recommendations 
            onNavigate={setCurrentScreen}
            skinTone={skinTone}
          />
        )}
        {currentScreen === 'profile' && (
          <Profile 
            onNavigate={setCurrentScreen}
          />
        )}
        {currentScreen === 'history' && (
          <History 
            onNavigate={setCurrentScreen}
          />
        )}
      </div>
    </div>
  );
}