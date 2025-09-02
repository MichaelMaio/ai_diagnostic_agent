// Import the createRoot API from the react-dom/client package.
import ReactDOM from 'react-dom/client';

// Import the main App component from the local file system.
import App from './App';

// Create a root container for rendering the React app inside the HTML element with id 'root'.
const root: ReactDOM.Root = ReactDOM.createRoot(document.getElementById('root')!);

// Render the App component inside the root container.
root.render(<App />);