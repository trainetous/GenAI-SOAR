
3. **TESTING.md**
```markdown
## Test Cases

### 1. Benign URL
- **Input**:
  - URL Length: Normal
  - SSL State: Trusted
  - Subdomains: None
  - All other features: False
- **Expected**: BENIGN verdict, no attribution

### 2. State-Sponsored APT
- **Input**:
  - Prefix/Suffix: True
  - SSL State: Trusted
  - Political Keywords: False
- **Expected**: MALICIOUS + "State-Sponsored" profile

### 3. Organized Cybercrime
- **Input**:
  - Shortened URL: True
  - Uses IP: True
  - Abnormal URL: True
- **Expected**: MALICIOUS + "Organized Cybercrime" profile

### 4. Hacktivist
- **Input**:
  - Political Keywords: True
  - Subdomains: Many
- **Expected**: MALICIOUS + "Hacktivist" profile
