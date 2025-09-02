import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  // Navigate to the Vending Machine application.
  await page.goto('http://localhost:5173/');
});

test('initial page load verification', async ({ page }) => {
  // Verify the page title and heading.
  await expect(page).toHaveTitle(/Vending Machine/);
  await expect(page.getByRole('heading')).toContainText('Vending Machine');

  // Verify the money display starts at $0.00 and the quarter image is visible.
  await expect(page.getByTestId('money-display')).toContainText('$0.00');
  await expect(page.getByTestId('quarter-image')).toBeVisible();

  // Verify the product table header.
  await expect(page.getByTestId('product-header')).toContainText('Product');
  await expect(page.getByTestId('price-header')).toContainText('Price');
  await expect(page.getByTestId('remaining-header')).toContainText('Remaining');

  // Verify the product table rows include the expected names, prices, and remaining counts, and that the product imges are visible.
  await expect(page.getByTestId('product-row-0').getByTestId('product-name')).toContainText('Coke');
  await expect(page.getByTestId('product-row-0').getByTestId('product-price')).toContainText('$1.25');
  await expect(page.getByTestId('product-row-0').getByTestId('product-remaining')).toContainText('5');
  await expect(page.getByTestId('product-row-0').getByTestId('product-image')).toBeVisible();
  await expect(page.getByTestId('product-row-1').getByTestId('product-name')).toContainText('Pepsi');
  await expect(page.getByTestId('product-row-1').getByTestId('product-price')).toContainText('$0.75');
  await expect(page.getByTestId('product-row-1').getByTestId('product-remaining')).toContainText('10');
  await expect(page.getByTestId('product-row-1').getByTestId('product-image')).toBeVisible();
});

test('insert money', async ({ page }) => {
  // Click on the quarter image.
  await page.getByTestId('quarter-image').click();

  // Verify the money display updates to show a quarter was inserted.
  await expect(page.getByTestId('money-display')).toContainText('$0.25');
});

test('purchase a product', async ({ page }) => {
  // Click on the quarter image enough times to buy 1 Coke ($1.25).
  for (let i = 0; i < 5; i++) {
    await page.getByTestId('quarter-image').click();
  }

  // Purchase 1 Coke by clicking on the product image.
  await page.getByTestId('product-row-0').getByTestId('product-image').click();

  // Verify there is no money remaining.
  await expect(page.getByTestId('money-display')).toContainText('$0.00');

  // Verify the number of Cokes remaining goes down by 1 (from 5 to 4).
  await expect(page.getByTestId('product-row-0').getByTestId('product-remaining')).toContainText('4');
});

test('failed purchase due to insufficient funds', async ({ page }) => {
  // Attempt to purchase a product even though there is no money in the machine.
  await page.getByTestId('product-row-0').getByTestId('product-image').click();

  // Verify the error message for insufficient funds.
  await expect(page.getByTestId('error-content')).toContainText('Insufficient funds. Coke costs $1.25.');
});

test('failed purchase due to insufficient inventory', async ({ page }) => {
  // Click on the quarter image enough times to buy 6 Cokes ($1.25 per Coke: 5 quarters * 6 Cokes = 30 quarters).
  for (let i = 0; i < 30; i++) {
    await page.getByTestId('quarter-image').click();
  }

  // Attempt to purchase 6 cokes, even though there are only 5 in stock.
  for (let i = 0; i < 6; i++) {
    await page.getByTestId('product-row-0').getByTestId('product-image').click();
  }
  
  // Verify the number of Cokes remaining is 0.
  await expect(page.getByTestId('product-row-0').getByTestId('product-remaining')).toContainText('0');

  // Verify there is enough money remaining for 1 Coke since we had enough money for 6 but could only buy 5.
  await expect(page.getByTestId('money-display')).toContainText("$1.25");

  // Verify the error message for failing to purchase the 6th Coke since it was out of stock.
  await expect(page.getByTestId('error-content')).toContainText('Sorry, Coke is out of stock');
});
