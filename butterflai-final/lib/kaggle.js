// lib/kaggle.js - REAL Kaggle API integration
export async function validateKaggleCredentials(username, key) {
  if (!username || !key) {
    return { valid: false, error: 'Kaggle username and key are required' }
  }
  
  try {
    // Test credentials by fetching user info
    const response = await fetch('https://www.kaggle.com/api/v1/account', {
      headers: {
        'Authorization': `Basic ${Buffer.from(`${username}:${key}`).toString('base64')}`
      }
    })
    
    if (!response.ok) {
      return { valid: false, error: 'Invalid Kaggle credentials' }
    }
    
    return { valid: true }
  } catch (error) {
    return { valid: false, error: error.message }
  }
}

export async function searchKaggleDatasets(query, username, key, limit = 10) {
  try {
    const response = await fetch(
      `https://www.kaggle.com/api/v1/datasets/list?search=${encodeURIComponent(query)}&sortBy=relevance&max=${limit}`,
      {
        headers: {
          'Authorization': `Basic ${Buffer.from(`${username}:${key}`).toString('base64')}`
        }
      }
    )
    
    if (!response.ok) {
      throw new Error('Failed to search Kaggle datasets')
    }
    
    const datasets = await response.json()
    
    return datasets.map(ds => ({
      id: ds.ref,
      name: ds.title,
      source: 'kaggle',
      size_gb: ds.totalBytes / (1024 * 1024 * 1024),
      num_images: ds.totalBytes ? Math.floor(ds.totalBytes / (1024 * 1024)) : 0, // Approx
      num_classes: ds.subtitle?.match(/\d+ classes?/i) ? 
        parseInt(ds.subtitle.match(/\d+/)[0]) : 0,
      relevance_pct: ds.score ? Math.min(100, ds.score * 10) : 50,
      license: ds.licenseName || 'Unknown',
      description: ds.description,
      format: detectKaggleFormat(ds),
      has_train_val_split: ds.description?.toLowerCase().includes('train') && 
                          ds.description?.toLowerCase().includes('val'),
      download_cmd: `kaggle datasets download -d ${ds.ref}`,
      url: ds.url,
      lastUpdated: ds.lastUpdated
    }))
  } catch (error) {
    console.error('Kaggle search failed:', error)
    throw error
  }
}

function detectKaggleFormat(dataset) {
  const desc = dataset.description?.toLowerCase() || ''
  const files = dataset.files || []
  
  if (desc.includes('image') || files.some(f => f.name.match(/\.(jpg|jpeg|png)$/i))) {
    return 'ImageFolder'
  } else if (desc.includes('csv') || files.some(f => f.name.endsWith('.csv'))) {
    return 'CSV'
  } else if (desc.includes('json')) {
    return 'JSON'
  }
  
  return 'Unknown'
}