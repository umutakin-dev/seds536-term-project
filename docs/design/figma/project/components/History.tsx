import { ArrowLeft, Calendar, ChevronRight, Trash2, Download, Sparkles } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface HistoryProps {
  onNavigate: (screen: 'home' | 'analysis' | 'recommendations' | 'profile' | 'history') => void;
}

interface AnalysisRecord {
  id: string;
  date: string;
  time: string;
  monkScale: number;
  monkColor: string;
  thumbnailUrl: string;
}

export function History({ onNavigate }: HistoryProps) {
  // Sample data - set to empty array for empty state
  const analysisHistory: AnalysisRecord[] = [
    {
      id: '1',
      date: 'Nov 8, 2025',
      time: '2:30 PM',
      monkScale: 6,
      monkColor: '#C09873',
      thumbnailUrl: 'https://images.unsplash.com/photo-1737978697863-5d65495b28ef?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b21hbiUyMG1lZGl1bSUyMHNraW4lMjB0b25lJTIwcG9ydHJhaXR8ZW58MXx8fHwxNzYyNjI3NjQ0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
    {
      id: '2',
      date: 'Nov 5, 2025',
      time: '10:15 AM',
      monkScale: 6,
      monkColor: '#C09873',
      thumbnailUrl: 'https://images.unsplash.com/photo-1737978697863-5d65495b28ef?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b21hbiUyMG1lZGl1bSUyMHNraW4lMjB0b25lJTIwcG9ydHJhaXR8ZW58MXx8fHwxNzYyNjI3NjQ0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
    {
      id: '3',
      date: 'Nov 1, 2025',
      time: '4:45 PM',
      monkScale: 5,
      monkColor: '#D5BA9D',
      thumbnailUrl: 'https://images.unsplash.com/photo-1737978697863-5d65495b28ef?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b21hbiUyMG1lZGl1bSUyMHNraW4lMjB0b25lJTIwcG9ydHJhaXR8ZW58MXx8fHwxNzYyNjI3NjQ0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
    {
      id: '4',
      date: 'Oct 28, 2025',
      time: '3:20 PM',
      monkScale: 6,
      monkColor: '#C09873',
      thumbnailUrl: 'https://images.unsplash.com/photo-1737978697863-5d65495b28ef?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b21hbiUyMG1lZGl1bSUyMHNraW4lMjB0b25lJTIwcG9ydHJhaXR8ZW58MXx8fHwxNzYyNjI3NjQ0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
    {
      id: '5',
      date: 'Oct 15, 2025',
      time: '11:00 AM',
      monkScale: 6,
      monkColor: '#C09873',
      thumbnailUrl: 'https://images.unsplash.com/photo-1737978697863-5d65495b28ef?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3b21hbiUyMG1lZGl1bSUyMHNraW4lMjB0b25lJTIwcG9ydHJhaXR8ZW58MXx8fHwxNzYyNjI3NjQ0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
    },
  ];

  // Group by month
  const groupedByMonth = analysisHistory.reduce((groups, record) => {
    const month = record.date.split(' ')[0] + ' ' + record.date.split(' ')[2]; // "Nov 2025"
    if (!groups[month]) {
      groups[month] = [];
    }
    groups[month].push(record);
    return groups;
  }, {} as Record<string, AnalysisRecord[]>);

  const isEmpty = analysisHistory.length === 0;

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-[#44475A] to-[#6272A4]">
      {/* Header */}
      <div className="flex items-center justify-between px-6 pt-14 pb-6">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => onNavigate('home')}
            className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center"
          >
            <ArrowLeft className="w-5 h-5 text-white" />
          </button>
          <h1 className="text-white">Analysis History</h1>
        </div>
        {!isEmpty && (
          <button className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center">
            <Calendar className="w-5 h-5 text-white" />
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 px-6 pb-6 overflow-y-auto">
        {isEmpty ? (
          /* Empty State */
          <div className="h-full flex flex-col items-center justify-center px-6 -mt-20">
            <div className="w-32 h-32 rounded-full bg-white/10 backdrop-blur-sm flex items-center justify-center mb-6">
              <Sparkles className="w-16 h-16 text-white/50" />
            </div>
            <h2 className="text-white text-center mb-3">No Analysis Yet</h2>
            <p className="text-white/70 text-center mb-8 max-w-xs">
              Start your first skin tone analysis to see your history here
            </p>
            <button
              onClick={() => onNavigate('analysis')}
              className="bg-gradient-to-r from-[#BD93F9] to-[#FF79C6] text-white px-8 py-4 rounded-[20px] transition-transform active:scale-95"
            >
              Start Analysis
            </button>
          </div>
        ) : (
          /* History List */
          <div className="space-y-6 pb-24">
            {Object.entries(groupedByMonth).map(([month, records]) => (
              <div key={month}>
                {/* Month Header */}
                <h3 className="text-white mb-4 px-2">{month}</h3>
                
                {/* Analysis Cards */}
                <div className="space-y-3">
                  {records.map((record) => (
                    <div
                      key={record.id}
                      className="relative bg-white/95 backdrop-blur-sm rounded-[32px] p-4 flex items-center gap-4 group"
                    >
                      {/* Thumbnail */}
                      <div className="w-16 h-16 rounded-[16px] overflow-hidden flex-shrink-0">
                        <ImageWithFallback
                          src={record.thumbnailUrl}
                          alt="Analysis photo"
                          className="w-full h-full object-cover"
                        />
                      </div>

                      {/* Info Section */}
                      <div className="flex-1 min-w-0">
                        <p className="text-gray-900 mb-1">{record.date}</p>
                        <div className="flex items-center gap-2">
                          <p className="text-sm text-gray-600">{record.time}</p>
                          <span className="text-xs text-gray-400">•</span>
                          <div className="flex items-center gap-2">
                            <div
                              className="w-5 h-5 rounded-full border-2 border-white shadow-sm"
                              style={{ backgroundColor: record.monkColor }}
                            />
                            <span className="text-sm text-gray-600">Monk {record.monkScale}</span>
                          </div>
                        </div>
                      </div>

                      {/* Right Arrow */}
                      <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />

                      {/* Swipe to Delete Hint */}
                      <div className="absolute right-0 top-0 bottom-0 w-16 bg-red-500 rounded-r-[32px] flex items-center justify-center opacity-0 group-hover:opacity-20 transition-opacity pointer-events-none">
                        <Trash2 className="w-5 h-5 text-white" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Export Button - Only show when there's history */}
      {!isEmpty && (
        <div className="absolute bottom-0 left-0 right-0 px-6 pb-8 pt-4 bg-gradient-to-t from-[#44475A] to-transparent">
          <button className="w-full bg-gradient-to-r from-[#BD93F9] to-[#FF79C6] text-white py-4 rounded-[20px] flex items-center justify-center gap-2 transition-transform active:scale-95">
            <Download className="w-5 h-5" />
            Export History
          </button>
        </div>
      )}
    </div>
  );
}
