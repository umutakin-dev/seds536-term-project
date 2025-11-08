import { ArrowLeft, User, Heart, Bell, Settings, HelpCircle, LogOut } from 'lucide-react';

interface ProfileProps {
  onNavigate: (screen: 'home' | 'analysis' | 'recommendations' | 'profile' | 'history') => void;
}

export function Profile({ onNavigate }: ProfileProps) {
  const menuItems = [
    { icon: User, label: 'Edit Profile', color: '#BD93F9' },
    { icon: Heart, label: 'Saved Products', color: '#FF79C6', badge: '12' },
    { icon: Bell, label: 'Notifications', color: '#BD93F9' },
    { icon: Settings, label: 'Settings', color: '#6272A4' },
    { icon: HelpCircle, label: 'Help & Support', color: '#BD93F9' },
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
        <h1 className="text-white">Profile</h1>
      </div>

      {/* Content */}
      <div className="flex-1 px-6 pb-6 overflow-y-auto">
        {/* Profile Card */}
        <div className="bg-white/95 backdrop-blur-sm rounded-[32px] p-6 mb-6 text-center">
          <div className="w-24 h-24 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center mx-auto mb-4">
            <User className="w-12 h-12 text-white" />
          </div>
          <h2 className="mb-1">Maya Johnson</h2>
          <p className="text-sm text-gray-600 mb-4">maya.j@email.com</p>
          
          <div className="grid grid-cols-3 gap-3 pt-4 border-t border-gray-200">
            <div>
              <p className="text-gray-900">Medium</p>
              <p className="text-xs text-gray-600 mt-1">Skin Tone</p>
            </div>
            <div>
              <p className="text-gray-900">7</p>
              <p className="text-xs text-gray-600 mt-1">Products</p>
            </div>
            <div>
              <p className="text-gray-900">32 days</p>
              <p className="text-xs text-gray-600 mt-1">Streak 🔥</p>
            </div>
          </div>
        </div>

        {/* Menu Items */}
        <div className="bg-white/95 backdrop-blur-sm rounded-[24px] p-3 mb-6">
          {menuItems.map((item, index) => (
            <button
              key={index}
              className="w-full flex items-center gap-4 p-4 rounded-[16px] hover:bg-gray-50 transition-colors"
            >
              <div 
                className="w-10 h-10 rounded-full flex items-center justify-center"
                style={{ backgroundColor: `${item.color}20` }}
              >
                <item.icon className="w-5 h-5" style={{ color: item.color }} />
              </div>
              <span className="flex-1 text-left text-gray-900">{item.label}</span>
              {item.badge && (
                <span className="px-2 py-1 bg-[#FF79C6] text-white text-xs rounded-full">
                  {item.badge}
                </span>
              )}
              <div className="w-2 h-2 rounded-full bg-gray-300" />
            </button>
          ))}
        </div>

        {/* Achievements */}
        <div className="bg-white/10 backdrop-blur-sm rounded-[24px] p-5 mb-6">
          <h3 className="text-white mb-4">Your Achievements</h3>
          <div className="flex gap-3 overflow-x-auto pb-2">
            <div className="flex-shrink-0 w-16 text-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center mb-2">
                <span className="text-2xl">🌟</span>
              </div>
              <p className="text-xs text-white">First Scan</p>
            </div>
            <div className="flex-shrink-0 w-16 text-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center mb-2">
                <span className="text-2xl">✨</span>
              </div>
              <p className="text-xs text-white">Routine Pro</p>
            </div>
            <div className="flex-shrink-0 w-16 text-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#BD93F9] to-[#FF79C6] flex items-center justify-center mb-2">
                <span className="text-2xl">💜</span>
              </div>
              <p className="text-xs text-white">Self Love</p>
            </div>
            <div className="flex-shrink-0 w-16 text-center">
              <div className="w-16 h-16 rounded-full bg-white/20 border-2 border-dashed border-white/50 flex items-center justify-center mb-2">
                <span className="text-2xl">🔒</span>
              </div>
              <p className="text-xs text-white/50">Locked</p>
            </div>
          </div>
        </div>

        {/* Logout Button */}
        <button className="w-full bg-white/10 backdrop-blur-sm text-white py-4 rounded-[20px] flex items-center justify-center gap-2 hover:bg-white/20 transition-colors">
          <LogOut className="w-5 h-5" />
          Log Out
        </button>
      </div>
    </div>
  );
}