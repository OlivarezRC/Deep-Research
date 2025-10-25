// Custom JavaScript for Chainlit app

// Function to find and modify the readme button
function waitForReadmeButton() {
  // Check if button exists already
  const readmeButton = document.getElementById('readme-button');
  
  if (readmeButton) {
    // Change the text to "Feedback"
    const span = readmeButton.querySelector('span');
    if (span && span.textContent.trim() === 'Readme') {
      span.textContent = 'Feedback';
    }

    // Only add the event listener once
    if (!readmeButton.dataset.feedbackListener) {
      readmeButton.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Open the feedback form in a new tab
        window.open('https://forms.office.com/r/0bwLp8VNYu', '_blank');
        
        return false;
      }, true);
      readmeButton.dataset.feedbackListener = 'true';
      console.log('Feedback button is ready');
    }
  } else {
    // If button doesn't exist yet, check again after a short delay
    setTimeout(waitForReadmeButton, 300);
  }
}

// Function to create and add the Work/Web toggle switch
function createWorkWebToggle() {
  // Check if toggle already exists
  if (document.querySelector('.work-web-toggle')) {
    return;
  }

  const header = document.getElementById('header');
  if (!header) {
    // If header doesn't exist yet, check again after a short delay
    setTimeout(createWorkWebToggle, 300);
    return;
  }

  // Create toggle container
  const toggleContainer = document.createElement('div');
  toggleContainer.className = 'header-toggle-container';

  // Create the toggle structure
  toggleContainer.innerHTML = `
    <div class="work-web-toggle">
      <div class="toggle-switch work" id="work-web-toggle">
        <span class="toggle-option work">Work</span>
        <span class="toggle-option web">Web</span>
        <div class="toggle-slider"></div>
      </div>
    </div>
  `;

  // Append to header
  header.appendChild(toggleContainer);

  // Add toggle functionality
  const toggleSwitch = document.getElementById('work-web-toggle');
  
  // Initialize currentMode from localStorage only
  const localMode = localStorage.getItem('workWebMode');
  
  let currentMode;
  if (localMode && (localMode === 'work' || localMode === 'web')) {
    currentMode = localMode;
    console.log(`Toggle initialized with local mode: ${currentMode}`);
  } else {
    currentMode = 'work';
    localStorage.setItem('workWebMode', currentMode);
    console.log('Toggle initialized with default mode: work');
  }
  
  // Set the initial visual state of the toggle
  toggleSwitch.className = `toggle-switch ${currentMode}`;
  document.body.setAttribute('data-mode', currentMode);

  toggleSwitch.addEventListener('click', function() {
    if (currentMode === 'work') {
      // Switch to web mode
      currentMode = 'web';
      toggleSwitch.className = 'toggle-switch web';
      console.log('Switched to Web mode');
      
      // Trigger any web mode functionality here
      handleModeChange('web');
    } else {
      // Switch to work mode
      currentMode = 'work';
      toggleSwitch.className = 'toggle-switch work';
      console.log('Switched to Work mode');
      
      // Trigger any work mode functionality here
      handleModeChange('work');
    }
  });

  console.log('Work/Web toggle created successfully');
}

// Function to send mode to Chainlit backend
async function sendModeToChainlit(mode) {
  console.log(`Sending mode to Chainlit: ${mode}`);
  
  // Method 1: Store in localStorage for SettingsBridge to pick up
  localStorage.setItem('pendingMode', mode);
  
  // Method 2: Send via postMessage to SettingsBridge
  window.postMessage(
    { source: "custom-js", type: "set-mode", value: mode },
    "*"
  );

  console.log(`Mode ${mode} communication initiated`);
}

// Function to handle mode changes
function handleModeChange(mode) {
  // Store the current mode in localStorage for persistence
  localStorage.setItem('workWebMode', mode);
  
  // Send to Chainlit using multiple methods
  sendModeToChainlit(mode);
  
  // Add custom event for other scripts to listen to
  const event = new CustomEvent('workWebModeChange', { 
    detail: { mode: mode } 
  });
  document.dispatchEvent(event);
  
  // You can add specific functionality for each mode here
  if (mode === 'work') {
    // Work mode specific functionality
    document.body.setAttribute('data-mode', 'work');
    console.log('Work mode activated and saved to Chainlit session');
  } else {
    // Web mode specific functionality
    document.body.setAttribute('data-mode', 'web');
    console.log('Web mode activated and saved to Chainlit session');
  }
}

// Function to initialize the toggle with stored preference
function initializeToggleState() {
  // Use localStorage only
  const localMode = localStorage.getItem('workWebMode');
  
  let storedMode;
  if (localMode && (localMode === 'work' || localMode === 'web')) {
    storedMode = localMode;
    console.log(`Using local storage mode: ${storedMode}`);
  } else {
    storedMode = 'work';
    localStorage.setItem('workWebMode', storedMode);
    console.log('Using default mode: work');
  }
  
  const toggleSwitch = document.getElementById('work-web-toggle');
  
  if (toggleSwitch) {
    toggleSwitch.className = `toggle-switch ${storedMode}`;
    handleModeChange(storedMode);
    console.log(`Initialized toggle to ${storedMode} mode`);
  } else {
    // Even if toggle doesn't exist yet, announce the stored mode
    announceInitialMode(storedMode);
  }
}

// Function to announce the initial mode to Chainlit
function announceInitialMode(mode) {
  // Send initial mode using the same method as handleModeChange
  sendModeToChainlit(mode);
  console.log(`Initial mode ${mode} announced to Chainlit`);
}

// Start looking for elements as soon as possible
function initializeApp() {
  waitForReadmeButton();
  createWorkWebToggle();
  
  // Initialize toggle state after a brief delay to ensure it's created
  setTimeout(initializeToggleState, 100);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}

// Also use a MutationObserver to catch dynamically added elements
const observer = new MutationObserver(function(mutations) {
  for (const mutation of mutations) {
    if (mutation.addedNodes.length) {
      waitForReadmeButton();
      createWorkWebToggle();
    }
  }
});

// Start observing once the DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    observer.observe(document.body, { childList: true, subtree: true });
  });
} else {
  observer.observe(document.body, { childList: true, subtree: true });
}
