import React from 'react'
import { useTranslation } from 'react-i18next'
import { VSCodeBadge } from '@vscode/webview-ui-toolkit/react'
import { formatLargeNumber } from '../../utils/formatters'
import { HistoryItem } from '../../types'

interface TransformIndicatorProps {
  isActive: boolean
  contextTokens: number
  actualTokenCount?: number
  className?: string
}

export const TransformIndicator: React.FC<TransformIndicatorProps> = ({
  isActive,
  contextTokens,
  actualTokenCount,
  className = ''
}) => {
  const { t } = useTranslation()
  
  // Calculate retention rate if transform is active
  const transformActive = isActive && actualTokenCount && actualTokenCount > contextTokens
  const retentionRate = transformActive 
    ? Math.round((contextTokens / (actualTokenCount || contextTokens)) * 100) 
    : 100
  
  // Calculate overage percentage if transform is active
  const overagePercent = transformActive
    ? Math.round(((actualTokenCount! - contextTokens) / contextTokens) * 100)
    : 0

  return (
    <VSCodeBadge
      className={`${isActive ? "bg-vscode-warningBackground text-vscode-warningForeground" : ""} ${className}`}
      title={
        isActive
          ? t("chat:tokenProgress.transformsEnabled", {
              contextTokens: formatLargeNumber(contextTokens),
              actualTokens: formatLargeNumber(actualTokenCount || contextTokens)
            }) + (transformActive
              ? "\n" + t("chat:tokenProgress.transformDetails", {
                  retentionRate,
                  overageRatio: overagePercent
                })
              : "")
          : t("chat:tokenProgress.transformsDisabled")
      }
    >
      <span className="flex items-center gap-1">
        <i className={`codicon codicon-${isActive ? "warning" : "check"} text-xs`} />
        {isActive ? (
          <>
            <span className="font-semibold">{t("chat:tokenProgress.transformActive")}</span>: {formatLargeNumber(contextTokens)} / {formatLargeNumber(actualTokenCount || contextTokens)}
          </>
        ) : t("chat:tokenProgress.transformInactive")}
      </span>
    </VSCodeBadge>
  )
}
