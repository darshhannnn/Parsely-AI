# 🌿 Contributing to Parsely AI

Thank you for your interest in contributing to Parsely AI! We welcome contributions that help make document processing fresher and more intelligent.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Docker (for containerized development)

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/parsely-ai.git
   cd parsely-ai
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your Gemini API key to .env
   ```

5. **Test Setup**
   ```bash
   python test_gemini_setup.py
   ```

## 🌱 Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small

### Commit Messages
Use the format: `type(scope): description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(parser): add support for HTML document processing
fix(api): resolve authentication token validation issue
docs(readme): update installation instructions
```

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates

## 🧪 Testing

### Run Tests
```bash
# Basic functionality test
python run_simple_test.py

# Full system test
python test_gemini_setup.py

# API tests
python test_api_direct.py
```

### Adding Tests
- Add unit tests for new functions
- Include integration tests for API endpoints
- Test with various document formats
- Verify error handling scenarios

## 📝 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add appropriate tests
   - Update documentation if needed

3. **Test Thoroughly**
   ```bash
   python test_gemini_setup.py
   python run_simple_test.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat(component): description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional document format support (RTF, ODT)
- [ ] Enhanced error handling and recovery
- [ ] Performance optimizations
- [ ] Security improvements
- [ ] UI/UX enhancements

### Medium Priority
- [ ] Additional LLM provider support
- [ ] Advanced caching mechanisms
- [ ] Monitoring and analytics
- [ ] Batch processing capabilities
- [ ] Multi-language support

### Documentation
- [ ] API documentation improvements
- [ ] Tutorial videos
- [ ] Use case examples
- [ ] Deployment guides
- [ ] Troubleshooting guides

## 🐛 Bug Reports

When reporting bugs, please include:
- **Environment**: OS, Python version, dependencies
- **Steps to reproduce**: Clear, step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Relevant error messages or logs
- **Screenshots**: If applicable

## 💡 Feature Requests

For feature requests, please provide:
- **Use case**: Why is this feature needed?
- **Description**: What should the feature do?
- **Examples**: How would it be used?
- **Priority**: How important is this feature?

## 🤝 Code of Conduct

### Our Pledge
We are committed to making participation in Parsely AI a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 📞 Getting Help

- **Documentation**: Check the README and docs
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special mentions for outstanding contributions

Thank you for helping make Parsely AI grow! 🌿