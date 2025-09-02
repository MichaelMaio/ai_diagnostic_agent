import { useState } from 'react';                     // React hook for managing component state.
import ProductList from './components/ProductList';   // Displays list of products.
import MoneySlot from './components/MoneySlot';       // Handles money insertion.
import ErrorMessage from './components/ErrorMessage'; // Displays error messages to the user.
import './styles.css';                                // Import shared CSS styles.

// Define initial product inventory for the vending machine.
const initialProducts = [
  { name: 'Coke', image: 'coke.png', price: 1.25, remaining: 5 },
  { name: 'Pepsi', image: 'pepsi.png', price: 0.75, remaining: 10 },
];

// Main App component.
const App = () => {

    // Setup showing error messages to the user.
    const [errorMessage, setErrorMessage] = useState("");

    function showError(message: string) {
        setErrorMessage(message);
    }

    // Track the total amount of money inserted by the user.
    const [amountInserted, setAmountInserted] = useState(0);

    // Track the current inventory of products.
    const [products, setProducts] = useState(initialProducts);

    // Handler for inserting a quarter into the vending machine.
    const handleInsertQuarter = () => {
        setAmountInserted(prev => prev - 0.25);

        // Clear any existing error message when a quarter is inserted.
        if (errorMessage) setErrorMessage("");
    };

    // Handler for purchasing a product.
    const handlePurchase = (index: number) => {

        // Get the selected product based on the index.
        const product = products[index];

        // Check if the product is out of stock.
        if (product.remaining === 0) {
            showError(`Sorry, ${product.name} is out of stock.`);
            return;
        }

        // Check if the user hasn't inserted enough money to purchase the product.
        if (amountInserted < product.price) {
            showError(`Insufficient funds. ${product.name} costs $${product.price}.`);
            return;
        }

        // Clear any existing error message when a valid purchase is made.
        if (errorMessage) setErrorMessage("");

        // Create a copy of the products array to avoid mutating state directly.
        const updatedProducts = [...products];

        // Decrease the product count by 1.
        updatedProducts[index].remaining -= 1;
    
        // Update the inventory state.
        setProducts(updatedProducts);

        // Deduct product price from the inserted amount.
        setAmountInserted(prev => (prev - product.price));
    };

    // Render the vending machine UI.
    return (
        <div className="vending-machine">
        <h1>Vending Machine</h1>

        {/* Render the money slot with insertion interaction */}
        <MoneySlot
          amountInserted={amountInserted}
          onInsertQuarter={handleInsertQuarter}
        />

        {/* Render the product list with purchase interaction */}
        <ProductList
          products={products}
          onPurchase={handlePurchase}
        />

        {errorMessage && (
          <ErrorMessage message={errorMessage} />
        )}

        </div>
    );
};

// Export the App component as default.
export default App;