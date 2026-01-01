# Frontend Changes: Dark/Light Mode Toggle

## Overview
Added a dark/light mode toggle button to the Course Materials Assistant application. The toggle is positioned in the top-right corner and allows users to switch between dark and light themes with smooth animations.

## Files Modified

### 1. `frontend/index.html`
- Added a theme toggle button with sun/moon SVG icons
- Button positioned fixed in top-right corner
- Includes proper ARIA labels for accessibility

### 2. `frontend/style.css`
Added the following CSS:

#### Light Theme CSS Variables (`[data-theme="light"]`)
Comprehensive light theme with WCAG AA compliant contrast:

- **Background colors**: Light and clean (`#f8fafc`, `#ffffff`)
- **Text colors**: Dark for good contrast (`#0f172a`, `#475569`)
- **Border colors**: Subtle but visible (`#cbd5e1`)
- **Primary colors**: Slightly adjusted for light backgrounds (`#1d4ed8`, `#1e40af`)
- **Shadows**: Lighter, more subtle shadows
- **Scrollbar colors**: Theme-aware scrollbar styling

#### Theme Toggle Button Styles (`.theme-toggle`)
- Fixed position, top-right (20px from edges)
- 44x44px circular button
- Hover effect with scale animation (1.1x)
- Focus ring for keyboard accessibility
- Icon visibility: Sun in dark mode, Moon in light mode

#### Transition Animations
- 0.3s smooth transitions on background-color, color, border-color
- Applied to body, containers, inputs, and UI elements
- Respects `prefers-reduced-motion` preference

#### Light Theme Specific Adjustments
- **Code blocks**: Light gray background with dark text
- **Source links**: Proper hover states for light backgrounds
- **Loading animation**: Adjusted dot colors
- **Error messages**: Red tints on white background
- **Success messages**: Green tints on white background
- **Blockquotes**: Blue left border with subtle background
- **Welcome message**: Light blue tinted background
- **Send button**: Adjusted shadow for visibility
- **Input placeholder**: Proper contrast color

#### Mobile Responsive
- Toggle button resized (40x40px) and repositioned (10px from edges) on mobile

### 3. `frontend/script.js`
Added the following JavaScript:

#### Early Theme Initialization (IIFE)
- Prevents flash of wrong theme on page load
- Checks localStorage for saved preference first
- Falls back to system preference (`prefers-color-scheme`)
- Defaults to dark theme if no preference

#### Theme Toggle Function (`toggleTheme()`)
- Toggles `data-theme` attribute on `<html>` element
- Saves preference to localStorage
- Updates ARIA label on button

#### ARIA Label Updater (`updateThemeToggleLabel()`)
- Dynamically updates button label based on current theme
- "Switch to light mode" / "Switch to dark mode"

## Features

### Design
- Circular 44x44px toggle button
- Sun icon (dark mode) / Moon icon (light mode)
- Smooth hover animation (scale 1.1)
- Focus ring for keyboard navigation
- Consistent with existing design aesthetic

### Accessibility
- Keyboard navigable (native `<button>` element)
- Dynamic ARIA labels
- Visible focus states
- Respects `prefers-reduced-motion` preference
- WCAG AA compliant color contrast in both themes

### Persistence
- Theme preference saved to localStorage
- Respects system preference as fallback
- No flash of wrong theme on page load

## Color Palette

### Dark Theme (Default)
```css
--background: #0f172a;        /* Very dark blue */
--surface: #1e293b;           /* Dark slate */
--surface-hover: #334155;     /* Medium slate */
--text-primary: #f1f5f9;      /* Light/white */
--text-secondary: #94a3b8;    /* Medium gray */
--border-color: #334155;      /* Medium slate */
--primary-color: #2563eb;     /* Blue */
--user-message: #2563eb;      /* Blue */
--assistant-message: #374151; /* Dark gray */
```

### Light Theme
```css
--background: #f8fafc;        /* Very light gray */
--surface: #ffffff;           /* Pure white */
--surface-hover: #f1f5f9;     /* Light gray */
--text-primary: #0f172a;      /* Very dark (high contrast) */
--text-secondary: #475569;    /* Medium dark gray */
--border-color: #cbd5e1;      /* Light gray border */
--primary-color: #1d4ed8;     /* Slightly darker blue */
--user-message: #2563eb;      /* Blue */
--assistant-message: #e2e8f0; /* Light gray */
```

### Light Theme Specific Colors
```css
/* Code blocks */
code background: #e2e8f0;
pre background: #f1f5f9;

/* Error states */
error background: #fef2f2;
error text: #dc2626;

/* Success states */
success background: #f0fdf4;
success text: #16a34a;

/* Welcome message */
welcome background: #eff6ff;
welcome border: #bfdbfe;
```

## Technical Implementation

### CSS Custom Properties
All colors use CSS variables defined in `:root` (dark) and `[data-theme="light"]` selectors. This allows:
- Single source of truth for colors
- Easy theme switching via `data-theme` attribute
- Consistent color usage across components

### Theme Switching Mechanism
1. `data-theme` attribute on `<html>` element
2. CSS variables automatically switch based on attribute
3. Smooth transitions applied for visual continuity
4. localStorage persists user preference

## Usage
Click the sun/moon icon in the top-right corner to toggle between dark and light modes. The preference is automatically saved and will persist across browser sessions.
