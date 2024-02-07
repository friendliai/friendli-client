# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""CLI App Settings."""


class Settings:
    """CLI app settings."""

    access_token_cookie_key = ""
    refresh_token_cookie_key = ""


class ProductionSettings:
    """Production CLI app settings."""

    access_token_cookie_key = "sAccessTokenProduction"
    refresh_token_cookie_key = "sRefreshTokenProduction"


class StagingSettings:
    """Staging CLI app settings."""

    access_token_cookie_key = "sAccessTokenStaging"
    refresh_token_cookie_key = "sRefreshTokenStaging"


class DevSettings:
    """Dev CLI app settings."""

    access_token_cookie_key = "sAccessTokenDev"
    refresh_token_cookie_key = "sRefreshTokenDev"
