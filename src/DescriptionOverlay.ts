/**
 * Creates a description overlay for examples that have a static description property
 * @param exampleClass - The example class to check for a description
 * @param container - The container element to append the overlay to
 */
export function createDescriptionOverlay(exampleClass: any, container: HTMLElement): void {
  // Check if the example class has a static description property
  if (exampleClass.description) {
    // Create the text overlay element
    const textOverlay = document.createElement('div');
    
    // Set the text content
    textOverlay.textContent = exampleClass.description;
    
    // Style the overlay
    Object.assign(textOverlay.style, {
      position: 'absolute',
      top: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      backgroundColor: 'rgba(40, 44, 52, 0.85)', 
      color: '#ffffff',
      padding: '12px 24px',
      borderRadius: '12px',
      fontFamily: 'Arial, sans-serif',
      fontSize: '16px',
      fontWeight: 'bold',
      textAlign: 'center',
      zIndex: '1000',
      boxShadow: '0 6px 16px rgba(0, 0, 0, 0.3), 0 0 0 2px rgba(255, 255, 255, 0.15)', 
      maxWidth: '80%',
      border: '2px solid rgba(255, 255, 255, 0.2)', 
      backdropFilter: 'blur(4px)', 
      WebkitBackdropFilter: 'blur(4px)', 
      letterSpacing: '0.5px',
      opacity: '1',
      transition: 'opacity 1s ease-out'
    });
    
    // Add to container
    container.style.position = 'relative';
    container.appendChild(textOverlay);
    
    // Set a timeout to fade out and remove the overlay after 5 seconds
    setTimeout(() => {
      // Start fade out
      textOverlay.style.opacity = '0';
      
      // Remove from DOM after fade completes
      setTimeout(() => {
        if (textOverlay.parentNode) {
          textOverlay.parentNode.removeChild(textOverlay);
        }
      }, 1000); // Wait for the 1s transition to complete
    }, 4000); // Start fade after 4 seconds (total visible time will be 5 seconds)
  }
}
