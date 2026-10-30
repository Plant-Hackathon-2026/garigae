import { useState } from 'react'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-950 to-slate-900 text-white flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full bg-slate-800/80 backdrop-blur-md rounded-2xl p-8 shadow-2xl border border-slate-700/60 text-center space-y-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-purple-500/10 border border-purple-500/20 rounded-full text-purple-300 text-sm font-medium">
          <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse"></span>
          React 19 + TypeScript + Tailwind CSS
        </div>

        <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
          Vite + React 환경 세팅 완료!
        </h1>

        <p className="text-slate-400 text-sm">
          Tailwind CSS가 정상적으로 적용되었습니다. 아래 버튼을 눌러 상태 변화를 테스트해보세요.
        </p>

        <div className="pt-2">
          <button
            type="button"
            onClick={() => setCount((prev) => prev + 1)}
            className="w-full py-3 px-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 active:scale-[0.98] transition-all rounded-xl font-semibold shadow-lg shadow-purple-600/30 cursor-pointer"
          >
            현재 클릭 횟수: <span className="font-mono font-bold text-lg">{count}</span>
          </button>
        </div>

        <div className="text-xs text-slate-500 pt-2 border-t border-slate-700/50">
          <code>src/App.tsx</code>를 수정하여 개발을 시작하세요.
        </div>
      </div>
    </div>
  )
}

export default App
