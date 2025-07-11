import flowbitePlugin from 'flowbite/plugin'

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './src/**/*.{html,js,svelte,ts}',
    './node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50:  '#8da5ea',
          100: '#5F82E8',
          200: '#5378e7',
          300: '#3763E7',
          400: '#2858e7',
          500: '#1449E6',
          600: '#103ec2',
          700: '#0736C1',
          800: '#06298F',
          900: '#052173',
        },
      },
    },
  },
  plugins: [flowbitePlugin],
}
