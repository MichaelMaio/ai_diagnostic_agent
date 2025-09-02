// Import shared css styles.
import '../styles.css';

// Props definition for MoneySlot component.
type Props = {
  amountInserted: number;        // Current amount of money inserted into the machine.
  onInsertQuarter: () => void;   // Callback function triggered when a quarter is inserted.
};

// MoneySlot component displays the current balance and handles coin insertion.
const MoneySlot = ({ amountInserted, onInsertQuarter }: Props) => {
  return (
    <div className="money-slot" data-testid="money-slot"> {/* Container for the money interface */}
      
      {/* Quarter image that users click to insert money */}
      <img
        src={new URL('../assets/money/images/quarter.png', import.meta.url).href} // Dynamically resolve image URL.
        alt="Insert quarter"                  // Accessible alt text for screen readers.
        onClick={onInsertQuarter}             // Trigger coin insertion when clicked.
        className="quarter-image"             // Style class for image sizing and positioning.
        data-testid="quarter-image"           // Test ID for automated UI testing.
      />

      {/* Display the current amount of money inserted */}
      <div className="money-display" data-testid="money-display">
        ${amountInserted.toFixed(2)}
      </div>
    </div>
  );
};

// Export the component for use in other parts of the application.
export default MoneySlot;