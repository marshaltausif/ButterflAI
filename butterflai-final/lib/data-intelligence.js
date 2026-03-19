// lib/data-intelligence.js - Complete dataset analysis
import { callAI } from './ai'

export async function analyzeDataset(dataset, apiKey, provider = 'gemini') {
  const analysis = {
    format: await detectFormat(dataset),
    structure: await analyzeStructure(dataset),
    quality: await assessQuality(dataset),
    metadata: await extractMetadata(dataset),
    recommendations: [],
    issues: []
  }

  // Generate preprocessing recommendations
  analysis.recommendations = await generateRecommendations(analysis, dataset)
  
  // Check for issues
  analysis.issues = await detectIssues(dataset, analysis)
  
  return analysis
}

async function detectFormat(dataset) {
  // For Kaggle/HF datasets, fetch metadata
  if (dataset.source === 'kaggle') {
    return await detectKaggleFormat(dataset)
  } else if (dataset.source === 'hf') {
    return await detectHFFormat(dataset)
  }
  
  return {
    type: 'unknown',
    formats: [],
    structure: 'unknown'
  }
}

async function detectKaggleFormat(dataset) {
  try {
    // Use Kaggle API to get dataset metadata
    const response = await fetch(
      `https://www.kaggle.com/api/v1/datasets/view/${dataset.id}`,
      { headers: { 'Authorization': `Bearer ${process.env.KAGGLE_API_KEY}` } }
    )
    
    if (!response.ok) throw new Error('Failed to fetch Kaggle metadata')
    
    const metadata = await response.json()
    
    // Analyze files to detect format
    const files = metadata.files || []
    const extensions = files.map(f => {
      const ext = f.name.split('.').pop().toLowerCase()
      return ext
    }).filter(Boolean)
    
    const uniqueExts = [...new Set(extensions)]
    
    // Detect format type
    let type = 'unknown'
    if (uniqueExts.some(ext => ['jpg', 'jpeg', 'png', 'webp'].includes(ext))) {
      type = 'image'
    } else if (uniqueExts.some(ext => ['csv', 'tsv', 'xlsx', 'parquet'].includes(ext))) {
      type = 'tabular'
    } else if (uniqueExts.some(ext => ['txt', 'json', 'xml'].includes(ext))) {
      type = 'text'
    } else if (uniqueExts.some(ext => ['mp3', 'wav', 'flac'].includes(ext))) {
      type = 'audio'
    } else if (uniqueExts.some(ext => ['mp4', 'avi', 'mov'].includes(ext))) {
      type = 'video'
    }
    
    // Detect structure
    let structure = 'unknown'
    if (metadata.subtitle?.includes('folder') || metadata.description?.includes('folder')) {
      structure = 'folder_by_class'
    } else if (files.some(f => f.name.includes('train') && f.name.includes('val'))) {
      structure = 'pre_split'
    } else if (files.some(f => f.name.endsWith('.csv'))) {
      structure = 'csv_with_paths'
    }
    
    return {
      type,
      formats: uniqueExts,
      structure,
      fileCount: files.length,
      totalSize: metadata.totalBytes,
      hasMetadata: true,
      metadata
    }
  } catch (error) {
    console.error('Kaggle format detection failed:', error)
    return {
      type: 'unknown',
      formats: [],
      structure: 'unknown',
      error: error.message
    }
  }
}

async function detectHFFormat(dataset) {
  try {
    // Use HuggingFace datasets info API
    const response = await fetch(
      `https://huggingface.co/api/datasets/${dataset.id}`
    )
    
    if (!response.ok) throw new Error('Failed to fetch HF metadata')
    
    const metadata = await response.json()
    
    // Detect format from HF tags and configs
    const tags = metadata.tags || []
    const cardData = metadata.cardData || {}
    
    let type = 'unknown'
    if (tags.includes('image-classification') || tags.includes('image')) {
      type = 'image'
    } else if (tags.includes('text-classification') || tags.includes('nlp')) {
      type = 'text'
    } else if (tags.includes('audio')) {
      type = 'audio'
    }
    
    return {
      type,
      formats: tags.filter(t => t.includes('format')).map(t => t.replace('format:', '')),
      structure: cardData.datasets?.[0]?.splits ? 'pre_split' : 'hf_dataset',
      fileCount: cardData.downloadSize,
      totalSize: cardData.downloadSize,
      hasMetadata: true,
      metadata
    }
  } catch (error) {
    console.error('HF format detection failed:', error)
    return {
      type: 'unknown',
      formats: [],
      structure: 'unknown',
      error: error.message
    }
  }
}

async function analyzeStructure(dataset) {
  // This would sample the actual dataset if accessible
  // For now, return based on metadata
  return {
    hasTrainValSplit: dataset.has_train_val_split || false,
    classCount: dataset.num_classes || 0,
    sampleCount: dataset.num_images || 0,
    classNames: dataset.classes || [],
    classDistribution: null, // Would be populated with actual counts
    averageSize: dataset.size_gb ? (dataset.size_gb * 1024) / (dataset.num_images || 1) : null
  }
}

async function assessQuality(dataset) {
  // Quality assessment based on metadata
  const issues = []
  const warnings = []
  
  // Check for common issues
  if (dataset.num_images && dataset.num_images < 1000) {
    warnings.push('Dataset is small (<1000 samples) - may need augmentation')
  }
  
  if (dataset.num_classes && dataset.num_images) {
    const avgPerClass = dataset.num_images / dataset.num_classes
    if (avgPerClass < 50) {
      warnings.push(`Very few samples per class (~${Math.round(avgPerClass)}) - high risk of overfitting`)
    }
  }
  
  if (dataset.license?.toLowerCase().includes('unknown')) {
    warnings.push('License unknown - verify usage rights')
  }
  
  return {
    score: calculateQualityScore(dataset, warnings.length),
    issues,
    warnings,
    hasCorruptedFiles: false, // Would need actual file check
    hasDuplicates: false,
    classBalance: 'unknown'
  }
}

function calculateQualityScore(dataset, warningCount) {
  let score = 100
  
  // Deduct for issues
  if (dataset.num_images && dataset.num_images < 1000) score -= 20
  if (!dataset.has_train_val_split) score -= 10
  if (warningCount > 0) score -= warningCount * 5
  
  return Math.max(0, Math.min(100, score))
}

async function extractMetadata(dataset) {
  return {
    id: dataset.id,
    name: dataset.name,
    source: dataset.source,
    license: dataset.license,
    description: dataset.description,
    url: dataset.source === 'kaggle' 
      ? `https://kaggle.com/datasets/${dataset.id}`
      : `https://huggingface.co/datasets/${dataset.id}`,
    citation: dataset.citation,
    version: dataset.version
  }
}

async function generateRecommendations(analysis, dataset) {
  const recommendations = []
  
  // Format-specific recommendations
  if (analysis.format.type === 'image') {
    recommendations.push({
      category: 'preprocessing',
      title: 'Standardize image sizes',
      description: 'Images should be resized to a consistent dimension for batching',
      action: 'resize',
      params: { target_size: '224x224' }
    })
    
    recommendations.push({
      category: 'augmentation',
      title: 'Apply data augmentation',
      description: 'Improve model generalization with augmentation',
      action: 'augment',
      params: {
        techniques: ['random_flip', 'random_rotate', 'color_jitter']
      }
    })
  } else if (analysis.format.type === 'tabular') {
    recommendations.push({
      category: 'preprocessing',
      title: 'Normalize numerical features',
      description: 'Scale features to similar ranges',
      action: 'normalize',
      params: { method: 'standardization' }
    })
    
    recommendations.push({
      category: 'encoding',
      title: 'Encode categorical variables',
      description: 'Convert categories to numerical format',
      action: 'encode',
      params: { method: 'one_hot' }
    })
  } else if (analysis.format.type === 'text') {
    recommendations.push({
      category: 'tokenization',
      title: 'Tokenize text',
      description: 'Convert text to tokens for model input',
      action: 'tokenize',
      params: { max_length: 512, truncation: true }
    })
  }
  
  // Quality-based recommendations
  if (analysis.quality.warnings?.length > 0) {
    recommendations.push({
      category: 'quality',
      title: 'Address quality warnings',
      description: analysis.quality.warnings.join(', '),
      action: 'review_quality',
      params: {}
    })
  }
  
  // Class balance recommendations
  if (analysis.structure.classCount > 0 && analysis.structure.sampleCount) {
    const avgPerClass = analysis.structure.sampleCount / analysis.structure.classCount
    if (avgPerClass < 100) {
      recommendations.push({
        category: 'balancing',
        title: 'Handle class imbalance',
        description: 'Consider using weighted loss or oversampling',
        action: 'balance_classes',
        params: { method: 'weighted_loss' }
      })
    }
  }
  
  return recommendations
}

async function detectIssues(dataset, analysis) {
  const issues = []
  
  // Critical issues
  if (!dataset.id) {
    issues.push({
      severity: 'critical',
      title: 'Invalid dataset',
      description: 'Dataset ID is missing'
    })
  }
  
  if (analysis.format.type === 'unknown') {
    issues.push({
      severity: 'critical',
      title: 'Unknown format',
      description: 'Could not detect dataset format. This dataset may not be compatible.'
    })
  }
  
  // Missing splits
  if (!dataset.has_train_val_split) {
    issues.push({
      severity: 'warning',
      title: 'Missing train/val splits',
      description: 'Dataset will be randomly split. Consider using a dataset with predefined splits for reproducibility.'
    })
  }
  
  // Small dataset
  if (dataset.num_images && dataset.num_images < 500) {
    issues.push({
      severity: 'warning',
      title: 'Very small dataset',
      description: `Only ${dataset.num_images} samples. Model may not generalize well.`
    })
  }
  
  return issues
}

export function generateIntelligenceReport(analysis) {
  return {
    summary: {
      formatType: analysis.format.type,
      totalSamples: analysis.structure.sampleCount,
      numClasses: analysis.structure.classCount,
      qualityScore: analysis.quality.score,
      hasIssues: analysis.issues.length > 0
    },
    details: analysis,
    recommendations: analysis.recommendations,
    issues: analysis.issues,
    timestamp: new Date().toISOString()
  }
}