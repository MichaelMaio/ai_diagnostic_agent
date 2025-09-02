// Import shared css styles.
import '../styles.css';

// Define the shape of a single product object.
type Product = {
  name: string;      // Product name, e.g., "Coke".
  price: number;     // Cost of the product.
  remaining: number; // Inventory count.
  image: string;     // Filename of the product image.
};

// Props accepted by the ProductList component.
type Props = {
  products: Product[];                    // Array of products to display.
  onPurchase: (index: number) => void;    // Callback to handle purchase logic.
};

// ProductList component renders a table of products.
const ProductList = ({ products, onPurchase }: Props) => {
  return (
    <table data-testid="product-table"> {/* Table element for displaying products */}
      <thead> {/* Table header row */}
        <tr>
          <th data-testid="product-header">Product</th>
          <th data-testid="price-header">Price</th>
          <th data-testid="remaining-header">Remaining</th>
        </tr>
      </thead>
      <tbody>
        {/* Iterate over the products array and render each product as a table row */}
        {products.map((product, index) => (
          <tr key={index} data-testid={`product-row-${index}`}>
            <td
              data-testid="product-name"  // Test ID for automated UI testing.   
              onClick={() => onPurchase(index)} // Handle purchase when product cell is clicked.
              className="clickable-product" // Style class to indicate the product is clickable.
            >
              {/* Display product image */}
              <img
                src={new URL(`../assets/products/images/${product.image}`, import.meta.url).href}   // Dynamically resolve image URL.
                alt={product.name}  // Accessible alt text for screen readers.
                data-testid="product-image" // Test ID for automated UI testing.
              />
              {/* Display product name next to image */}
              {product.name}
            </td>
            {/* Show price */}
            <td data-testid="product-price">${product.price}</td>
            {/* Show remaining inventory */}
            <td data-testid="product-remaining">{product.remaining}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// Export the component to be used in other parts of the application.
export default ProductList;