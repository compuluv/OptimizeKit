import React, { useState, useEffect } from 'react';
import { 
  Key, 
  Zap, 
  TrendingUp, 
  Users, 
  Settings, 
  Plus,
  Copy,
  Eye,
  EyeOff,
  Activity,
  BarChart3,
  Brain,
  Sparkles
} from 'lucide-react';

const Dashboard = () => {
  const [apiKeys, setApiKeys] = useState([
    { id: '1', name: 'Production API', key: 'ok_live_abc123...', plan: 'pro', usage: 15420, limit: 50000, created: '2024-01-15' },
    { id: '2', name: 'Development API', key: 'ok_test_def456...', plan: 'free', usage: 2341, limit: 10000, created: '2024-02-01' }
  ]);
  
  const [showKeys, setShowKeys] = useState({});
  const [activeTab, setActiveTab] = useState('overview');

  const stats = {
    totalRequests: 17761,
    optimizationRate: 94.2,
    avgResponseTime: 145,
    activeProjects: 12
  };

  const recentOptimizations = [
    { id: 1, prompt: 'Explain quantum computing', strategy: 'clarity', improvement: '+23%', time: '2 min ago' },
    { id: 2, prompt: 'Write marketing copy', strategy: 'creative', improvement: '+31%', time: '5 min ago' },
    { id: 3, prompt: 'Debug this code', strategy: 'technical', improvement: '+18%', time: '12 min ago' },
    { id: 4, prompt: 'Summarize document', strategy: 'concise', improvement: '+27%', time: '18 min ago' }
  ];

  const toggleKeyVisibility = (keyId) => {
    setShowKeys(prev => ({ ...prev, [keyId]: !prev[keyId] }));
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // In a real app, you'd show a toast notification here
  };

  const generateNewKey = () => {
    const newKey = {
      id: (apiKeys.length + 1).toString(),
      name: `API Key ${apiKeys.length + 1}`,
      key: `ok_live_${Math.random().toString(36).substr(2, 12)}...`,
      plan: 'free',
      usage: 0,
      limit: 10000,
      created: new Date().toISOString().split('T')[0]
    };
    setApiKeys([...apiKeys, newKey]);
  };

  const TabButton = ({ id, label, icon: Icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
        activeTab === id 
          ? 'bg-blue-600 text-white' 
          : 'text-gray-600 hover:bg-gray-100'
      }`}
    >
      <Icon size={18} />
      {label}
    </button>
  );

  const StatCard = ({ title, value, subtitle, icon: Icon, trend }) => (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-lg ${trend === 'up' ? 'bg-green-100' : 'bg-blue-100'}`}>
          <Icon className={`${trend === 'up' ? 'text-green-600' : 'text-blue-600'}`} size={24} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Sparkles className="text-white" size={24} />
              </div>
              <h1 className="text-xl font-bold text-gray-900">OptimizeKit</h1>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-600">
                Welcome back, <span className="font-medium">Alex</span>
              </div>
              <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg">
                <Settings size={20} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Navigation Tabs */}
        <div className="flex gap-2 mb-8">
          <TabButton id="overview" label="Overview" icon={BarChart3} />
          <TabButton id="api-keys" label="API Keys" icon={Key} />
          <TabButton id="optimize" label="Optimize" icon={Brain} />
          <TabButton id="analytics" label="Analytics" icon={TrendingUp} />
        </div>

        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard
                title="Total Requests"
                value={stats.totalRequests.toLocaleString()}
                subtitle="This month"
                icon={Activity}
                trend="up"
              />
              <StatCard
                title="Optimization Rate"
                value={`${stats.optimizationRate}%`}
                subtitle="Success rate"
                icon={TrendingUp}
                trend="up"
              />
              <StatCard
                title="Avg Response Time"
                value={`${stats.avgResponseTime}ms`}
                subtitle="Last 24h"
                icon={Zap}
              />
              <StatCard
                title="Active Projects"
                value={stats.activeProjects}
                subtitle="This month"
                icon={Users}
              />
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Optimizations</h3>
              <div className="space-y-4">
                {recentOptimizations.map((item) => (
                  <div key={item.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div className="flex-1">
                      <p className="font-medium text-gray-900">{item.prompt}</p>
                      <div className="flex items-center gap-4 mt-1">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {item.strategy}
                        </span>
                        <span className="text-sm text-gray-600">{item.time}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-sm font-medium bg-green-100 text-green-800">
                        {item.improvement}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'api-keys' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold text-gray-900">API Keys</h2>
              <button
                onClick={generateNewKey}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Plus size={18} />
                Generate New Key
              </button>
            </div>

            <div className="grid gap-6">
              {apiKeys.map((apiKey) => (
                <div key={apiKey.id} className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">{apiKey.name}</h3>
                      <p className="text-sm text-gray-600">Created on {apiKey.created}</p>
                    </div>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      apiKey.plan === 'pro' 
                        ? 'bg-purple-100 text-purple-800' 
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {apiKey.plan.toUpperCase()}
                    </span>
                  </div>

                  <div className="flex items-center gap-3 mb-4">
                    <div className="flex-1 font-mono text-sm bg-gray-100 p-3 rounded-lg">
                      {showKeys[apiKey.id] ? apiKey.key : apiKey.key.replace(/./g, '•')}
                    </div>
                    <button
                      onClick={() => toggleKeyVisibility(apiKey.id)}
                      className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                    >
                      {showKeys[apiKey.id] ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                    <button
                      onClick={() => copyToClipboard(apiKey.key)}
                      className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                    >
                      <Copy size={18} />
                    </button>
                  </div>

                  <div className="flex justify-between items-center">
                    <div className="text-sm text-gray-600">
                      Usage: <span className="font-medium">{apiKey.usage.toLocaleString()}</span> / {apiKey.limit.toLocaleString()} requests
                    </div>
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          (apiKey.usage / apiKey.limit) > 0.8 ? 'bg-red-600' : 'bg-blue-600'
                        }`}
                        style={{ width: `${Math.min((apiKey.usage / apiKey.limit) * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'optimize' && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Prompt Optimizer</h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Original Prompt
                  </label>
                  <textarea
                    placeholder="Enter your prompt here..."
                    className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Strategy
                      </label>
                      <select className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                        <option value="clarity">Clarity</option>
                        <option value="concise">Concise</option>
                        <option value="creative">Creative</option>
                        <option value="technical">Technical</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Target Model
                      </label>
                      <select className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                        <option value="gpt-4">GPT-4</option>
                        <option value="claude-3-opus">Claude 3 Opus</option>
                        <option value="gemini-pro">Gemini Pro</option>
                      </select>
                    </div>
                  </div>
                  
                  <button className="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    Optimize Prompt
                  </button>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Optimized Result
                  </label>
                  <div className="h-32 p-3 bg-gray-50 border border-gray-300 rounded-lg text-gray-500">
                    Optimized prompt will appear here...
                  </div>
                  
                  <div className="mt-4 p-4 bg-green-50 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-green-800">Confidence Score</span>
                      <span className="text-sm font-bold text-green-800">---%</span>
                    </div>
                    <div className="w-full bg-green-200 rounded-full h-2">
                      <div className="bg-green-600 h-2 rounded-full w-0 transition-all duration-500" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900">Analytics</h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Usage Trends</h3>
                <div className="h-64 flex items-center justify-center text-gray-500">
                  Chart placeholder - Usage over time
                </div>
              </div>
              
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Optimization Strategies</h3>
                <div className="space-y-3">
                  {['Clarity', 'Creative', 'Technical', 'Concise'].map((strategy, index) => (
                    <div key={strategy} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">{strategy}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${Math.random() * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-gray-900">
                          {Math.floor(Math.random() * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;