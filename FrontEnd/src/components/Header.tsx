import React from "react"

const Header: React.FC = () => {
  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <a href="/" className="flex items-center gap-3" aria-label="healthpilot.ai home">
          <img
            src="/images/healthpilot-logo.jpg"
            alt="healthpilot.ai logo"
            className="h-8 w-auto"
            loading="eager"
            width={128}
            height={32}
          />
          <span className="text-lg font-semibold tracking-tight">healthpilot.ai</span>
        </a>
        <div className="hidden sm:flex items-center gap-3 text-sm text-muted-foreground">
          <span className="hidden md:inline">HIPAA Compliant</span>
          <span>•</span>
          <span className="hidden md:inline">SOC 2 Certified</span>
          <span>•</span>
          <span className="hidden md:inline">99.9% Uptime</span>
        </div>
      </div>
    </header>
  )
}

export default Header
