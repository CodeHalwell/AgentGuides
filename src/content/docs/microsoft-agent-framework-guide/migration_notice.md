---
title: "⚠️ MIGRATION NOTICE"
description: "Date: November 12, 2025 Status: This directory is superseded by new platform-specific guides"
framework: microsoft-agent-framework
---

# ⚠️ MIGRATION NOTICE
## This Guide Has Been Split Into Platform-Specific Versions

**Date:** November 12, 2025  
**Status:** This directory is superseded by new platform-specific guides

---

## 🎯 Important: Use New Guides

This **original combined guide** has been **split into two separate, focused guides** for better developer experience:

### **For Python Developers**
📁 **Use:** `./python/`

**Why:** 
- ✅ 100% Python-focused content
- ✅ No .NET distractions
- ✅ Python-specific best practices
- ✅ asyncio/await patterns
- ✅ pytest examples

**Start Here:**
```bash
cd python
# Read GUIDE_INDEX.md for navigation
```

### **For .NET Developers**
📁 **Use:** `./dotnet/`

**Why:**
- ✅ 100% .NET/C#-focused content
- ✅ No Python distractions
- ✅ .NET-specific best practices
- ✅ TPL/Task patterns
- ✅ xUnit examples

**Start Here:**
```bash
cd dotnet
# Read GUIDE_INDEX.md for navigation
```

---

## 📚 What Changed?

### **Original Structure (This Directory)**
```
Microsoft_Agent_Framework_Guide/
├── README.md                                    [Mixed Python + .NET]
├── microsoft_agent_framework_comprehensive_guide.md [Mixed]
├── microsoft_agent_framework_recipes.md         [Mixed]
├── microsoft_agent_framework_production_guide.md [Mixed]
└── microsoft_agent_framework_diagrams.md        [Mixed]
```

**Problem:** Mixed code examples caused confusion

### **New Structure (Platform-Specific)**
```
python/          [100% Python]
├── GUIDE_INDEX.md                               [NEW - Navigation hub]
├── README.md                                    [Python-focused]
├── microsoft_agent_framework_python_comprehensive_guide.md
├── microsoft_agent_framework_python_recipes.md
├── microsoft_agent_framework_python_production_guide.md
└── microsoft_agent_framework_python_diagrams.md

dotnet/          [100% .NET]
├── GUIDE_INDEX.md                               [NEW - Navigation hub]
├── README.md                                    [.NET-focused]
├── microsoft_agent_framework_dotnet_comprehensive_guide.md
├── microsoft_agent_framework_dotnet_recipes.md
├── microsoft_agent_framework_dotnet_production_guide.md
└── microsoft_agent_framework_dotnet_diagrams.md
```

**Solution:** Separated guides for clarity

---

## 🔍 Quick Comparison

| Aspect | Original (Here) | New Python Guide | New .NET Guide |
|--------|----------------|------------------|----------------|
| **Target Audience** | Both platforms | Python devs only | .NET devs only |
| **Code Examples** | Mixed | 100% Python | 100% C# |
| **Installation** | Both | pip/venv | dotnet/NuGet |
| **Testing** | Mixed | pytest | xUnit/NUnit |
| **Best Practices** | Generic | Python-specific | .NET-specific |
| **Navigation** | Basic | GUIDE_INDEX | GUIDE_INDEX |
| **Confusion** | Some | None | None |

---

## 📖 Documentation Map

### **If You're Learning Python:**
1. Start: `./python/GUIDE_INDEX.md`
2. Setup: `./python/README.md`
3. Learn: `./python/microsoft_agent_framework_python_comprehensive_guide.md`
4. Practice: `./python/microsoft_agent_framework_python_recipes.md`

### **If You're Learning .NET:**
1. Start: `./dotnet/GUIDE_INDEX.md`
2. Setup: `./dotnet/README.md`
3. Learn: `./dotnet/microsoft_agent_framework_dotnet_comprehensive_guide.md`
4. Practice: `./dotnet/microsoft_agent_framework_dotnet_recipes.md`

---

## ⚡ Why This Change?

### **Benefits of Split Guides**

1. **Faster Learning**
   - No need to filter through irrelevant code
   - Direct path to what you need
   - Platform-specific best practices

2. **Better Examples**
   - Copy-paste ready code
   - No syntax confusion
   - Idiomatic patterns for each language

3. **Comprehensive Coverage**
   - Each guide is complete for its platform
   - No compromises in depth
   - Full deployment guidance

4. **Easier Navigation**
   - GUIDE_INDEX for quick reference
   - Clear learning paths
   - Searchable topic index

---

## 🚀 Migration Instructions

### **For Existing Users**

1. **Identify Your Platform**
   - Using Python? → Go to Python guide
   - Using .NET? → Go to .NET guide
   - Using both? → Use both guides

2. **Bookmark New Location**
   - Update your bookmarks
   - Update documentation links
   - Share new paths with team

3. **Optional: Archive This Directory**
   ```bash
   # This directory can be safely archived or removed
   mv Microsoft_Agent_Framework_Guide Microsoft_Agent_Framework_Guide_ARCHIVED
   ```

---

## 📋 What Stays Here?

This original directory contains:
- ✅ Original mixed-platform documentation (for reference)
- ✅ This migration notice
- ⚠️ No longer actively maintained

**Recommendation:** Use new platform-specific guides

---

## ❓ FAQ

### **Q: Can I still use this directory?**
A: Yes, but we strongly recommend using the new split guides for better experience.

### **Q: Is content missing from the new guides?**
A: No, both new guides are comprehensive. They contain all original content plus enhancements.

### **Q: What about diagrams?**
A: Diagrams are included in both guides (they're mostly platform-agnostic).

### **Q: Should I delete this directory?**
A: You can archive it. The new guides are the official documentation.

### **Q: Are the new guides up-to-date?**
A: Yes, they were created November 2025 and validated with Context7 and GitHub.

---

## 📞 Need Help?

### **Resources**
- **Summary:** See `../MICROSOFT_AGENT_FRAMEWORK_SPLIT_SUMMARY.md`
- **Python Guide:** `./python/`
- **. NET Guide:** `./dotnet/`
- **GitHub:** https://github.com/microsoft/agent-framework

### **Questions?**
- Check GUIDE_INDEX in your platform's guide
- Review comprehensive guide for concepts
- Reference recipes for examples

---

## ✅ Action Required

### **Choose Your Path:**

#### **Python Developer**
```bash
cd python
open GUIDE_INDEX.md
```

#### **.NET Developer**
```bash
cd dotnet
open GUIDE_INDEX.md
```

---

**This directory is now legacy. Please use the new platform-specific guides above. 🚀**

---

**Migration Date:** November 12, 2025  
**New Guides Created:** November 12, 2025  
**Status:** ⚠️ Superseded - Use new guides

