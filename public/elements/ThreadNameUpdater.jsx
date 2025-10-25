// public/elements/ThreadNameUpdater.jsx
// Custom element that updates thread name via PUT request to /project/thread

import { useEffect } from 'react';

export default function ThreadNameUpdater() {
  // Access userInput from global props object (Chainlit pattern)
  const userInput = props?.userInput;
  
  useEffect(() => {
    const updateThreadName = async () => {
      try {
        // Extract thread ID from current URL
        const currentUrl = window.location.href;
        const threadIdMatch = currentUrl.match(/\/thread\/([a-f0-9-]+)/);
        
        if (!threadIdMatch) {
          console.warn('[ThreadNameUpdater] Thread ID not found in URL:', currentUrl);
          return;
        }
        
        const threadId = threadIdMatch[1];
        
        // Truncate userInput to a reasonable length for thread name (max 100 chars)
        const threadName = userInput.length > 100 
          ? userInput.substring(0, 97) + '...' 
          : userInput;
        
        // Send PUT request to update thread name
        const response = await fetch('/project/thread', {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            threadId: threadId,
            name: threadName
          })
        });
        
        const results = await response.json();
        console.log('[ThreadNameUpdater] results:', results);
        
      } catch (error) {
        console.error('[ThreadNameUpdater] Error updating thread name:', error);
      }
    };
    
    // Only update if userInput is provided and is not empty
    if (userInput && userInput.trim()) {
      console.log('[ThreadNameUpdater] Updating thread name to:', userInput);
      // Add a small delay to ensure the page is fully loaded
      setTimeout(updateThreadName, 500);
    }
    
  }, [userInput]);
  
  return null; // This component doesn't render anything visible
}