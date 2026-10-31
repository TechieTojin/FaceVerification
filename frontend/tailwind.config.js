/** @type {import('tailwindcss').Config} */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        page: v('page'),
        surface: v('surface'),
        'surface-2': v('surface-2'),
        line: v('line'),
        ink: v('ink'),
        'ink-2': v('ink-2'),
        muted: v('muted'),
        primary: v('primary'),
        'primary-strong': v('primary-strong'),
        'primary-soft': v('primary-soft'),
        good: v('good'),
        'good-text': v('good-text'),
        warning: v('warning'),
        serious: v('serious'),
        critical: v('critical'),
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgb(0 0 0 / 0.04), 0 0 0 1px rgb(var(--line) / 1)',
        pop: '0 12px 32px -8px rgb(0 0 0 / 0.25), 0 0 0 1px rgb(var(--line) / 1)',
      },
      keyframes: {
        'fade-in': { from: { opacity: '0', transform: 'translateY(4px)' }, to: { opacity: '1', transform: 'none' } },
        pulse2: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.35' } },
      },
      animation: {
        'fade-in': 'fade-in 160ms ease-out',
        pulse2: 'pulse2 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
