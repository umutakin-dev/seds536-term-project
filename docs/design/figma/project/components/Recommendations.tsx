import { ArrowLeft, Heart, ShoppingBag, Sparkles, Star } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface RecommendationsProps {
  onNavigate: (screen: 'home' | 'analysis' | 'recommendations' | 'profile' | 'history') => void;
  skinTone: string | null;
}

export function Recommendations({ onNavigate, skinTone }: RecommendationsProps) {
  const products = [
    {
      name: 'Hydrating Serum',
      brand: 'GlowUp',
      category: 'Serum',
      rating: 4.8,
      price: '$32',
      match: '98%',
      image: 'https://images.unsplash.com/photo-1693146187444-d3d993a74a53?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxiZWF1dHklMjBwcm9kdWN0cyUyMG1pbmltYWx8ZW58MXx8fHwxNzYyNjIxMzA2fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
    {
      name: 'Vitamin C Brightening',
      brand: 'Pure Radiance',
      category: 'Treatment',
      rating: 4.9,
      price: '$45',
      match: '96%',
      image: 'https://images.unsplash.com/photo-1609357912334-e96886c0212b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxza2luY2FyZSUyMHJvdXRpbmUlMjBuYXR1cmFsfGVufDF8fHx8MTc2MjU3MzQyNXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
    {
      name: 'Gentle Cleanser',
      brand: 'SkinLove',
      category: 'Cleanser',
      rating: 4.7,
      price: '$24',
      match: '95%',
      image: 'https://images.unsplash.com/photo-1693146187444-d3d993a74a53?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxiZWF1dHklMjBwcm9kdWN0cyUyMG1pbmltYWx8ZW58MXx8fHwxNzYyNjIxMzA2fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
    {
      name: 'Night Repair Cream',
      brand: 'Dream Skin',
      category: 'Moisturizer',
      rating: 4.9,
      price: '$58',
      match: '94%',
      image: 'https://images.unsplash.com/photo-1609357912334-e96886c0212b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxza2luY2FyZSUyMHJvdXRpbmUlMjBuYXR1cmFsfGVufDF8fHx8MTc2MjU3MzQyNXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
  ];

  const routineSteps = [
    { step: 'Morning', time: '5 min', products: 3 },
    { step: 'Evening', time: '7 min', products: 4 },
    { step: 'Weekly', time: '15 min', products: 2 },
  ];

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-[#44475A] to-[#6272A4]">
      {/* Header */}
      <div className="flex items-center gap-4 px-6 pt-14 pb-6">
        <button 
          onClick={() => onNavigate('home')}
          className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center"
        >
          <ArrowLeft className="w-5 h-5 text-white" />
        </button>
        <div className="flex-1">
          <h1 className="text-white">Your Recommendations</h1>
          {skinTone && (
            <p className="text-white/70 text-sm mt-1">Matched for {skinTone} skin tone</p>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 px-6 pb-6 overflow-y-auto">
        {/* Skin Match Card */}
        <div className="bg-white/95 backdrop-blur-sm rounded-[32px] p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h2>Personalized Just For You</h2>
              <p className="text-sm text-gray-600">Based on your unique skin analysis</p>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-3">
            {routineSteps.map((routine) => (
              <div key={routine.step} className="bg-gradient-to-br from-[#BD93F9]/10 to-[#FF79C6]/10 rounded-[16px] p-3 text-center">
                <p className="text-gray-900 text-sm mb-1">{routine.step}</p>
                <p className="text-xs text-gray-600">{routine.time}</p>
                <p className="text-xs text-[#BD93F9] mt-1">{routine.products} products</p>
              </div>
            ))}
          </div>
        </div>

        {/* Products */}
        <div className="mb-6">
          <h3 className="text-white mb-4">Recommended Products</h3>
          <div className="space-y-4">
            {products.map((product, index) => (
              <div 
                key={index}
                className="bg-white/95 backdrop-blur-sm rounded-[24px] p-4 flex gap-4"
              >
                <div className="w-20 h-20 rounded-[16px] overflow-hidden bg-gray-100 flex-shrink-0">
                  <ImageWithFallback 
                    src={product.image}
                    alt={product.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <div className="flex-1 min-w-0">
                      <p className="text-gray-900 truncate">{product.name}</p>
                      <p className="text-sm text-gray-600">{product.brand}</p>
                    </div>
                    <button className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0">
                      <Heart className="w-4 h-4 text-gray-600" />
                    </button>
                  </div>
                  <div className="flex items-center gap-2 mb-2">
                    <div className="flex items-center gap-1">
                      <Star className="w-3 h-3 text-yellow-500 fill-current" />
                      <span className="text-sm text-gray-700">{product.rating}</span>
                    </div>
                    <span className="text-xs text-gray-400">•</span>
                    <span className="text-sm text-gray-700">{product.price}</span>
                    <span className="ml-auto text-sm text-[#BD93F9]">{product.match} match</span>
                  </div>
                  <button className="w-full bg-gradient-to-r from-[#BD93F9] to-[#FF79C6] text-white py-2 rounded-[12px] text-sm flex items-center justify-center gap-2 transition-transform active:scale-95">
                    <ShoppingBag className="w-4 h-4" />
                    Add to Routine
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Tips Section */}
        <div className="bg-white/10 backdrop-blur-sm rounded-[24px] p-5 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-5 h-5 text-[#FF79C6]" />
            <h3 className="text-white">Pro Tip for Your Skin Tone</h3>
          </div>
          <p className="text-white/90 text-sm leading-relaxed">
            {skinTone === 'Deep' || skinTone === 'Tan' 
              ? "Look for products with vitamin C and niacinamide to enhance your natural glow. SPF is essential for protecting against hyperpigmentation."
              : "Gentle formulas with hyaluronic acid work wonderfully. Always use SPF 30+ to protect your delicate complexion."}
          </p>
        </div>
      </div>
    </div>
  );
}