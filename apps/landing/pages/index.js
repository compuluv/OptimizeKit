import Head from 'next/head'

export default function Home() {
  return (
    <>
      <Head>
        <title>OptimizeKit - AI Prompt Optimization Platform</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <div>
        {/* Navigation */}
        <nav className="fixed w-full z-50 bg-gray-900/90 backdrop-blur-sm border-b border-gray-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold">OK</span>
                </div>
                <span className="text-xl font-bold text-white">OptimizeKit</span>
              </div>
              <div className="hidden md:flex items-center space-x-8">
                <a href="#features" className="text-gray-300 hover:text-white transition-colors">Features</a>
                <a href="#pricing" className="text-gray-300 hover:text-white transition-colors">Pricing</a>
                <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors text-white">
                  Get Started
                </button>
              </div>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <main className="bg-gray-900 text-white">
          <section className="relative min-h-screen flex items-center justify-center">
            <div className="relative z-10 max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
              <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
                Optimize Your
                <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent block">
                  AI Prompts
                </span>
                <span className="text-4xl md:text-6xl">Effortlessly</span>
              </h1>
              
              <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
                Transform your AI interactions with intelligent prompt optimization, 
                smart model selection, and personalized responses. 
                <span className="text-blue-400 font-semibold">94% improvement rate</span> guaranteed.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <button className="bg-blue-600 hover:bg-blue-700 px-8 py-4 rounded-xl text-lg font-semibold transition-all transform hover:scale-105">
                  Start Free Trial
                </button>
                <button className="border border-gray-600 hover:border-gray-500 px-8 py-4 rounded-xl text-lg font-semibold transition-all transform hover:scale-105">
                  View Demo
                </button>
              </div>
            </div>
          </section>

          {/* Features Preview */}
          <section className="py-20 px-4 sm:px-6 lg:px-8">
            <div className="max-w-6xl mx-auto text-center">
              <h2 className="text-4xl font-bold mb-16">
                Powerful Features for 
                <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent"> AI Excellence</span>
              </h2>
              <div className="grid md:grid-cols-3 gap-8">
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700">
                  <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
                    <span className="text-white font-bold text-2xl">⚡</span>
                  </div>
                  <h3 className="text-2xl font-bold mb-4">Smart Optimization</h3>
                  <p className="text-gray-300">AI-powered prompt enhancement with 94%+ improvement rate</p>
                </div>
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700">
                  <div className="w-16 h-16 bg-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
                    <span className="text-white font-bold text-2xl">🎯</span>
                  </div>
                  <h3 className="text-2xl font-bold mb-4">Model Selection</h3>
                  <p className="text-gray-300">Intelligent model recommendation across GPT, Claude, and Gemini</p>
                </div>
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700">
                  <div className="w-16 h-16 bg-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
                    <span className="text-white font-bold text-2xl">👤</span>
                  </div>
                  <h3 className="text-2xl font-bold mb-4">Personalization</h3>
                  <p className="text-gray-300">Adaptive prompts that learn from user behavior and preferences</p>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </>
  )
}
