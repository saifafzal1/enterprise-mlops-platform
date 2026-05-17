"""
Dataset builder for Jira user story → test script pairs.
Generates synthetic data when real pairs are not yet available.
Real data (560 pairs) plugged in via --real-cypress / --real-playwright args.
"""
import argparse, json, os, random
from pathlib import Path

CYPRESS_TEMPLATES = [
    {
        "story": "As a user, I want to log in with valid credentials so I can access my account.",
        "acceptance_criteria": ["Login page loads", "Valid email and password accepted", "Redirect to dashboard"],
        "script": """describe('Login', () => {
  it('should login with valid credentials', () => {
    cy.visit('/login');
    cy.get('[data-testid="email"]').type('user@example.com');
    cy.get('[data-testid="password"]').type('Password123!');
    cy.get('[data-testid="submit"]').click();
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="welcome-msg"]').should('be.visible');
  });
});"""
    },
    {
        "story": "As a user, I want to see an error message when I enter invalid credentials.",
        "acceptance_criteria": ["Error shown on wrong password", "User stays on login page"],
        "script": """describe('Login Error', () => {
  it('should show error on invalid credentials', () => {
    cy.visit('/login');
    cy.get('[data-testid="email"]').type('wrong@example.com');
    cy.get('[data-testid="password"]').type('wrongpass');
    cy.get('[data-testid="submit"]').click();
    cy.get('[data-testid="error-msg"]').should('contain', 'Invalid credentials');
    cy.url().should('include', '/login');
  });
});"""
    },
    {
        "story": "As a user, I want to search for products by keyword.",
        "acceptance_criteria": ["Search bar visible", "Results appear after search", "Results match keyword"],
        "script": """describe('Product Search', () => {
  it('should return results for keyword search', () => {
    cy.visit('/products');
    cy.get('[data-testid="search-input"]').type('laptop');
    cy.get('[data-testid="search-btn"]').click();
    cy.get('[data-testid="product-list"]').should('be.visible');
    cy.get('[data-testid="product-item"]').should('have.length.greaterThan', 0);
    cy.get('[data-testid="product-item"]').first().should('contain', 'laptop');
  });
});"""
    },
    {
        "story": "As a user, I want to add a product to the shopping cart.",
        "acceptance_criteria": ["Add to cart button visible", "Cart count increments", "Item appears in cart"],
        "script": """describe('Shopping Cart', () => {
  it('should add product to cart', () => {
    cy.visit('/products/1');
    cy.get('[data-testid="add-to-cart"]').click();
    cy.get('[data-testid="cart-count"]').should('contain', '1');
    cy.visit('/cart');
    cy.get('[data-testid="cart-item"]').should('have.length', 1);
  });
});"""
    },
    {
        "story": "As a user, I want to complete the checkout process with valid payment details.",
        "acceptance_criteria": ["Checkout form visible", "Payment accepted", "Order confirmation shown"],
        "script": """describe('Checkout', () => {
  it('should complete checkout with valid payment', () => {
    cy.visit('/checkout');
    cy.get('[data-testid="card-number"]').type('4111111111111111');
    cy.get('[data-testid="expiry"]').type('12/26');
    cy.get('[data-testid="cvv"]').type('123');
    cy.get('[data-testid="place-order"]').click();
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="order-id"]').should('not.be.empty');
  });
});"""
    },
    {
        "story": "As a user, I want to filter products by category.",
        "acceptance_criteria": ["Category filter visible", "Products update after filter", "Active filter shown"],
        "script": """describe('Product Filter', () => {
  it('should filter products by category', () => {
    cy.visit('/products');
    cy.get('[data-testid="category-electronics"]').click();
    cy.get('[data-testid="active-filter"]').should('contain', 'Electronics');
    cy.get('[data-testid="product-item"]').each(($el) => {
      cy.wrap($el).find('[data-testid="category-label"]').should('contain', 'Electronics');
    });
  });
});"""
    },
    {
        "story": "As a user, I want to view my order history.",
        "acceptance_criteria": ["Order history page accessible", "Past orders listed", "Order details viewable"],
        "script": """describe('Order History', () => {
  it('should display order history', () => {
    cy.visit('/account/orders');
    cy.get('[data-testid="order-list"]').should('be.visible');
    cy.get('[data-testid="order-item"]').should('have.length.greaterThan', 0);
    cy.get('[data-testid="order-item"]').first().click();
    cy.get('[data-testid="order-detail"]').should('be.visible');
  });
});"""
    },
    {
        "story": "As a user, I want to reset my password via email.",
        "acceptance_criteria": ["Reset link sent on valid email", "Success message shown"],
        "script": """describe('Password Reset', () => {
  it('should send reset email for valid account', () => {
    cy.visit('/forgot-password');
    cy.get('[data-testid="email-input"]').type('user@example.com');
    cy.get('[data-testid="send-reset"]').click();
    cy.get('[data-testid="success-msg"]').should('contain', 'Check your email');
  });
});"""
    },
    {
        "story": "As an admin, I want to create a new user account.",
        "acceptance_criteria": ["Admin can fill new user form", "User saved to system", "Confirmation message shown"],
        "script": """describe('Admin Create User', () => {
  it('should create a new user successfully', () => {
    cy.visit('/admin/users/new');
    cy.get('[data-testid="user-name"]').type('Jane Doe');
    cy.get('[data-testid="user-email"]').type('jane@example.com');
    cy.get('[data-testid="user-role"]').select('editor');
    cy.get('[data-testid="save-user"]').click();
    cy.get('[data-testid="toast-success"]').should('contain', 'User created');
  });
});"""
    },
    {
        "story": "As a user, I want to update my profile information.",
        "acceptance_criteria": ["Profile form editable", "Changes saved", "Updated info shown"],
        "script": """describe('Profile Update', () => {
  it('should update profile information', () => {
    cy.visit('/profile/edit');
    cy.get('[data-testid="display-name"]').clear().type('New Name');
    cy.get('[data-testid="phone"]').clear().type('+1234567890');
    cy.get('[data-testid="save-profile"]').click();
    cy.get('[data-testid="profile-name"]').should('contain', 'New Name');
  });
});"""
    },
]

PLAYWRIGHT_TEMPLATES = [
    {
        "story": "As a user, I want to log in with valid credentials so I can access my account.",
        "acceptance_criteria": ["Login page loads", "Valid credentials accepted", "Redirect to dashboard"],
        "script": """const { test, expect } = require('@playwright/test');

test('should login with valid credentials', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[data-testid="email"]', 'user@example.com');
  await page.fill('[data-testid="password"]', 'Password123!');
  await page.click('[data-testid="submit"]');
  await expect(page).toHaveURL(/dashboard/);
  await expect(page.locator('[data-testid="welcome-msg"]')).toBeVisible();
});"""
    },
    {
        "story": "As a user, I want to see an error message when I enter invalid credentials.",
        "acceptance_criteria": ["Error shown on wrong password", "User stays on login page"],
        "script": """const { test, expect } = require('@playwright/test');

test('should show error on invalid credentials', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[data-testid="email"]', 'wrong@example.com');
  await page.fill('[data-testid="password"]', 'wrongpass');
  await page.click('[data-testid="submit"]');
  await expect(page.locator('[data-testid="error-msg"]')).toContainText('Invalid credentials');
  await expect(page).toHaveURL(/login/);
});"""
    },
    {
        "story": "As a user, I want to search for products by keyword.",
        "acceptance_criteria": ["Search bar visible", "Results appear after search", "Results match keyword"],
        "script": """const { test, expect } = require('@playwright/test');

test('should return results for keyword search', async ({ page }) => {
  await page.goto('/products');
  await page.fill('[data-testid="search-input"]', 'laptop');
  await page.click('[data-testid="search-btn"]');
  await expect(page.locator('[data-testid="product-list"]')).toBeVisible();
  const items = page.locator('[data-testid="product-item"]');
  await expect(items).toHaveCountGreaterThan(0);
  await expect(items.first()).toContainText('laptop');
});"""
    },
    {
        "story": "As a user, I want to add a product to the shopping cart.",
        "acceptance_criteria": ["Add to cart button visible", "Cart count increments", "Item appears in cart"],
        "script": """const { test, expect } = require('@playwright/test');

test('should add product to cart', async ({ page }) => {
  await page.goto('/products/1');
  await page.click('[data-testid="add-to-cart"]');
  await expect(page.locator('[data-testid="cart-count"]')).toContainText('1');
  await page.goto('/cart');
  await expect(page.locator('[data-testid="cart-item"]')).toHaveCount(1);
});"""
    },
    {
        "story": "As a user, I want to complete the checkout process with valid payment details.",
        "acceptance_criteria": ["Checkout form visible", "Payment accepted", "Order confirmation shown"],
        "script": """const { test, expect } = require('@playwright/test');

test('should complete checkout with valid payment', async ({ page }) => {
  await page.goto('/checkout');
  await page.fill('[data-testid="card-number"]', '4111111111111111');
  await page.fill('[data-testid="expiry"]', '12/26');
  await page.fill('[data-testid="cvv"]', '123');
  await page.click('[data-testid="place-order"]');
  await expect(page.locator('[data-testid="order-confirmation"]')).toBeVisible();
  await expect(page.locator('[data-testid="order-id"]')).not.toBeEmpty();
});"""
    },
    {
        "story": "As a user, I want to filter products by category.",
        "acceptance_criteria": ["Category filter visible", "Products update after filter", "Active filter shown"],
        "script": """const { test, expect } = require('@playwright/test');

test('should filter products by category', async ({ page }) => {
  await page.goto('/products');
  await page.click('[data-testid="category-electronics"]');
  await expect(page.locator('[data-testid="active-filter"]')).toContainText('Electronics');
  const items = page.locator('[data-testid="product-item"]');
  const count = await items.count();
  for (let i = 0; i < count; i++) {
    await expect(items.nth(i).locator('[data-testid="category-label"]')).toContainText('Electronics');
  }
});"""
    },
    {
        "story": "As a user, I want to view my order history.",
        "acceptance_criteria": ["Order history accessible", "Past orders listed", "Order details viewable"],
        "script": """const { test, expect } = require('@playwright/test');

test('should display order history', async ({ page }) => {
  await page.goto('/account/orders');
  await expect(page.locator('[data-testid="order-list"]')).toBeVisible();
  const orders = page.locator('[data-testid="order-item"]');
  await expect(orders).toHaveCountGreaterThan(0);
  await orders.first().click();
  await expect(page.locator('[data-testid="order-detail"]')).toBeVisible();
});"""
    },
    {
        "story": "As a user, I want to reset my password via email.",
        "acceptance_criteria": ["Reset link sent on valid email", "Success message shown"],
        "script": """const { test, expect } = require('@playwright/test');

test('should send reset email for valid account', async ({ page }) => {
  await page.goto('/forgot-password');
  await page.fill('[data-testid="email-input"]', 'user@example.com');
  await page.click('[data-testid="send-reset"]');
  await expect(page.locator('[data-testid="success-msg"]')).toContainText('Check your email');
});"""
    },
    {
        "story": "As an admin, I want to create a new user account.",
        "acceptance_criteria": ["Admin fills new user form", "User saved", "Confirmation shown"],
        "script": """const { test, expect } = require('@playwright/test');

test('should create a new user successfully', async ({ page }) => {
  await page.goto('/admin/users/new');
  await page.fill('[data-testid="user-name"]', 'Jane Doe');
  await page.fill('[data-testid="user-email"]', 'jane@example.com');
  await page.selectOption('[data-testid="user-role"]', 'editor');
  await page.click('[data-testid="save-user"]');
  await expect(page.locator('[data-testid="toast-success"]')).toContainText('User created');
});"""
    },
    {
        "story": "As a user, I want to update my profile information.",
        "acceptance_criteria": ["Profile form editable", "Changes saved", "Updated info shown"],
        "script": """const { test, expect } = require('@playwright/test');

test('should update profile information', async ({ page }) => {
  await page.goto('/profile/edit');
  await page.fill('[data-testid="display-name"]', 'New Name');
  await page.fill('[data-testid="phone"]', '+1234567890');
  await page.click('[data-testid="save-profile"]');
  await expect(page.locator('[data-testid="profile-name"]')).toContainText('New Name');
});"""
    },
]


def _augment(pairs: list, target: int, framework: str) -> list:
    """Augment by paraphrasing story/criteria until target count reached."""
    result = list(pairs)
    prefixes = ["As a registered user,", "As an authenticated user,", "As a customer,",
                "As a member,", "As an end user,"]
    verbs = ["I want to", "I need to", "I would like to", "I should be able to"]
    random.seed(42)
    while len(result) < target:
        base = random.choice(pairs)
        new = dict(base)
        story = base["story"]
        for p in prefixes:
            if story.startswith("As a"):
                story = p + story[story.index(","):]
                break
        new = {"story": story,
               "acceptance_criteria": base["acceptance_criteria"],
               "framework": framework,
               "script": base["script"]}
        result.append(new)
    return result[:target]


def build(cypress_out, playwright_out, cypress_test, playwright_test,
          real_cypress=None, real_playwright=None, n_synthetic=280, test_split=0.15):

    def load_real(path, framework):
        if path and os.path.exists(path):
            import jsonlines
            with jsonlines.open(path) as r:
                return list(r)
        return []

    def make_pairs(templates, framework):
        pairs = [{"story": t["story"],
                  "acceptance_criteria": t["acceptance_criteria"],
                  "framework": framework,
                  "script": t["script"]} for t in templates]
        return _augment(pairs, n_synthetic, framework)

    for framework, templates, real_path, out_train, out_test in [
        ("cypress",    CYPRESS_TEMPLATES,    real_cypress,    cypress_out,    cypress_test),
        ("playwright", PLAYWRIGHT_TEMPLATES, real_playwright, playwright_out, playwright_test),
    ]:
        real = load_real(real_path, framework)
        pairs = real if len(real) >= n_synthetic else make_pairs(templates, framework)

        random.shuffle(pairs)
        split = int(len(pairs) * (1 - test_split))
        train_pairs, test_pairs = pairs[:split], pairs[split:]

        Path(out_train).parent.mkdir(parents=True, exist_ok=True)
        import jsonlines
        with jsonlines.open(out_train, mode="w") as w:
            for p in train_pairs:
                w.write(p)
        with jsonlines.open(out_test, mode="w") as w:
            for p in test_pairs:
                w.write(p)
        print(f"{framework}: {len(train_pairs)} train / {len(test_pairs)} test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cypress-out");    ap.add_argument("--playwright-out")
    ap.add_argument("--cypress-test");   ap.add_argument("--playwright-test")
    ap.add_argument("--real-cypress",  default=None)
    ap.add_argument("--real-playwright", default=None)
    ap.add_argument("--n-synthetic", type=int, default=280)
    ap.add_argument("--test-split",  type=float, default=0.15)
    args = ap.parse_args()
    build(args.cypress_out, args.playwright_out, args.cypress_test, args.playwright_test,
          args.real_cypress, args.real_playwright, args.n_synthetic, args.test_split)
