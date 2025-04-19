import React from 'react'
import { useTranslation } from 'react-i18next'
import { VSCodeDivider } from '@vscode/webview-ui-toolkit/react'
import { formatLargeNumber } from '../../utils/formatters'
import { SettingToggle } from './SettingToggle'
import { SettingSection } from './SettingSection'
import { SettingDescription } from './SettingDescription'
import { SettingTitle } from './SettingTitle'
import { HistoryItem } from '../../types'

interface TransformSettingsProps {
  isOpenRouter: boolean
  openRouterTransformsEnabled: boolean
  onToggleTransforms: (enabled: boolean) => void
  currentTaskItem?: HistoryItem
  contextTokens: number
}

export const TransformSettings: React.FC<TransformSettingsProps> = ({
  isOpenRouter,
  openRouterTransformsEnabled,
  onToggleTransforms,
  currentTaskItem,
  contextTokens
}) => {
  const { t } = useTranslation()

  if (!isOpenRouter) {
    return null
  }

  const actualTokenCount = currentTaskItem?.actualTokenCount || contextTokens
  const transformActive = openRouterTransformsEnabled && actualTokenCount > contextTokens
  
  // Calculate retention rate if transform is active
  const retentionRate = transformActive 
    ? Math.round((contextTokens / actualTokenCount) * 100) 
    : 100
  
  // Calculate overage percentage if transform is active
  const overagePercent = transformActive
    ? Math.round(((actualTokenCount - contextTokens) / contextTokens) * 100)
    : 0

  return (
    <>
      <VSCodeDivider />
      <SettingSection>
        <SettingTitle>
          {t('settings:openrouter.transforms.title')}
          {transformActive && (
            <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-vscode-warningBackground text-vscode-warningForeground">
              {t('settings:openrouter.transforms.active')}
            </span>
          )}
        </SettingTitle>
        <SettingDescription>
          {t('settings:openrouter.transforms.description')}
        </SettingDescription>
        
        <div className="mt-2">
          <SettingToggle
            id="openrouter-transforms-toggle"
            checked={openRouterTransformsEnabled}
            onChange={onToggleTransforms}
            label={t('settings:openrouter.transforms.toggle')}
          />
        </div>
        
        {transformActive && (
          <div className="mt-3 p-3 bg-vscode-editorWidget rounded-md">
            <h4 className="text-sm font-semibold mb-1">
              {t('settings:openrouter.transforms.status')}
            </h4>
            <ul className="text-xs space-y-1">
              <li className="flex justify-between">
                <span>{t('settings:openrouter.transforms.actualTokens')}:</span>
                <span className="font-mono">{formatLargeNumber(actualTokenCount)}</span>
              </li>
              <li className="flex justify-between">
                <span>{t('settings:openrouter.transforms.transformedTokens')}:</span>
                <span className="font-mono">{formatLargeNumber(contextTokens)}</span>
              </li>
              <li className="flex justify-between">
                <span>{t('settings:openrouter.transforms.retentionRate')}:</span>
                <span className="font-mono">{retentionRate}%</span>
              </li>
              <li className="flex justify-between">
                <span>{t('settings:openrouter.transforms.overagePercent')}:</span>
                <span className="font-mono">{overagePercent}%</span>
              </li>
            </ul>
          </div>
        )}
      </SettingSection>
    </>
  )
}
