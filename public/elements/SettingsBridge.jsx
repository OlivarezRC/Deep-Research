// public/elements/SettingsBridge.jsx
// Bridge component that listens for postMessage events and communicates with Chainlit backend

import { useEffect } from 'react';

export default function SettingsBridge() {
  useEffect(() => {
    const handleMessage = async (evt) => {
      const data = evt?.data || {};
      
      if ((data?.source === "custom-js" || data?.source === "mode-checker") && data?.type === "set-mode") {
        try {
          // Use the global callAction API provided by Chainlit
          const result = await callAction({
            name: "set_mode",
            payload: { mode: data.value }
          });
          
          if (result.success) {
            console.log(`Mode successfully set to: ${data.value}`);
            // Clear the pending mode after successful setting
            localStorage.removeItem('pendingMode');
          } else {
            console.warn(`Failed to set mode to: ${data.value}`);
            // Keep it in localStorage for retry
            localStorage.setItem('pendingMode', data.value);
          }
          
        } catch (error) {
          console.error("Error in SettingsBridge:", error);
          // Fallback: Store in localStorage for retry
          localStorage.setItem('pendingMode', data.value);
        }
      }
    };

    // Check for pending mode in localStorage on component mount
    const checkPendingMode = () => {
      const pendingMode = localStorage.getItem('pendingMode');
      const localMode = localStorage.getItem('workWebMode');
      console.log('Checking pending mode in SettingsBridge:', pendingMode);
      console.log('Checking local mode in SettingsBridge:', localMode);

      if ((pendingMode && (pendingMode === 'work' || pendingMode === 'web')) || (localMode && (localMode === 'work' || localMode === 'web'))) {
        // Re-trigger the mode setting
        window.postMessage({
          source: "mode-checker",
          type: "set-mode",
          value: pendingMode || localMode
        }, "*");
        console.log('Re-triggering pending mode from SettingsBridge:', pendingMode);
      }
    };

    window.addEventListener("message", handleMessage);
    console.log("SettingsBridge: Message listener attached");
    
    // Check for pending mode immediately when component mounts
    setTimeout(checkPendingMode, 100);
    
    // Cleanup function
    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, []);
  
  return null;
}