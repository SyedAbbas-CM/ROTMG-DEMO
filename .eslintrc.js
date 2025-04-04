module.exports = {
  root: true,
  env: {
    browser: true,
    es2021: true,
    node: true
  },
  extends: 'eslint:recommended',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  rules: {
    'import/no-unresolved': 'off', // Disable case sensitivity checks for imports
    'import/case-sensitivity': 'off', // Disable case sensitivity checks
    'import/no-duplicates': 'off' // Disable duplicate import checks
  }
}; 